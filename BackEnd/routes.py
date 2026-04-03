import json
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from models import SessionCreate, SessionResponse, ChatRequest, ChatResponse
from database import get_collection
from bson import ObjectId
from datetime import datetime

import nn_client

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    nn_ok = await nn_client.check_health()
    return {
        "status": "ok",
        "nn_service": "connected" if nn_ok else "unavailable",
    }

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: SessionCreate):
    system_prompt = await nn_client.get_system_prompt()

    session = {
        "user_id": body.user_id,
        "created_at": datetime.utcnow(),
        "messages": [
            {"role": "system", "content": system_prompt, "timestamp": datetime.utcnow().isoformat()},
        ],
    }

    collection = get_collection("sessions")
    result = await collection.insert_one(session)

    return SessionResponse(
        id=str(result.inserted_id),
        user_id=body.user_id,
        created_at=session["created_at"],
        message_count=0,
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(user_id: str = "anonymous"):
    collection = get_collection("sessions")
    sessions = []
    async for doc in collection.find({"user_id": user_id}).sort("created_at", -1):
        user_messages = [m for m in doc["messages"] if m["role"] != "system"]
        sessions.append(SessionResponse(
            id=str(doc["_id"]),
            user_id=doc["user_id"],
            created_at=doc["created_at"],
            message_count=len(user_messages),
        ))
    return sessions


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    collection = get_collection("sessions")
    try:
        doc = await collection.find_one({"_id": ObjectId(session_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [m for m in doc["messages"] if m["role"] != "system"]
    return {"session_id": session_id, "messages": messages}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    collection = get_collection("sessions")
    try:
        result = await collection.delete_one({"_id": ObjectId(session_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest):
    print(f"[CHAT] Received message for session {session_id}: {body.message[:50]}...")
    collection = get_collection("sessions")
    try:
        doc = await collection.find_one({"_id": ObjectId(session_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = {"role": "user", "content": body.message, "timestamp": datetime.utcnow().isoformat()}
    doc["messages"].append(user_msg)

    messages_for_model = [{"role": m["role"], "content": m["content"]} for m in doc["messages"]]
    print(f"[CHAT] Sending {len(messages_for_model)} messages to NN service...")
    try:
        response_text = await nn_client.generate_response(messages_for_model)
        print(f"[CHAT] Got response: {response_text[:80]}...")
    except Exception as e:
        print(f"[CHAT] NN service error: {e}")
        logger.error(f"NN service error: {e}")
        raise HTTPException(status_code=503, detail=f"Model service error: {str(e)}")

    assistant_msg = {"role": "assistant", "content": response_text, "timestamp": datetime.utcnow().isoformat()}
    doc["messages"].append(assistant_msg)

    await collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$push": {"messages": {"$each": [user_msg, assistant_msg]}}},
    )

    return ChatResponse(response=response_text, session_id=session_id)


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    collection = get_collection("sessions")

    try:
        doc = await collection.find_one({"_id": ObjectId(session_id)})
    except Exception:
        await websocket.send_json({"error": "Invalid session ID"})
        await websocket.close()
        return

    if not doc:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                user_text = payload.get("message", "").strip()
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if not user_text:
                await websocket.send_json({"error": "Empty message"})
                continue

            doc = await collection.find_one({"_id": ObjectId(session_id)})

            user_msg = {"role": "user", "content": user_text, "timestamp": datetime.utcnow().isoformat()}
            doc["messages"].append(user_msg)

            messages_for_model = [{"role": m["role"], "content": m["content"]} for m in doc["messages"]]

            try:
                response_text = await nn_client.generate_response(messages_for_model)
            except Exception as e:
                await websocket.send_json({"error": f"Model error: {str(e)}"})
                continue

            assistant_msg = {"role": "assistant", "content": response_text, "timestamp": datetime.utcnow().isoformat()}
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$push": {"messages": {"$each": [user_msg, assistant_msg]}}},
            )
            await websocket.send_json({
                "response": response_text,
                "session_id": session_id,
            })

    except WebSocketDisconnect:
        pass
