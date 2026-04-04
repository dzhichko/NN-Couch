import json
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status, Depends
from models import (
    SessionCreate, SessionResponse, ChatRequest, ChatResponse,
    UserRegister, UserLogin, TokenResponse, UserResponse
)
from database import get_collection
from bson import ObjectId
from datetime import datetime
from auth import hash_password, verify_password, create_access_token, get_current_user
import nn_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ============ АУТЕНТИФИКАЦИЯ ============

@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """Регистрация нового пользователя"""
    collection = get_collection("users")

    # Проверяем, не существует ли пользователь с таким email или username
    existing_user = await collection.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username}
        ]
    })

    if existing_user:
        if existing_user["email"] == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
        else:
            raise HTTPException(status_code=400, detail="Username already taken")

    # Хэшируем пароль
    hashed_password = hash_password(user_data.password)

    # Создаем пользователя
    user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True,
        "sessions": []  # для хранения ID сессий чата
    }

    result = await collection.insert_one(user_doc)
    user_id = str(result.inserted_id)

    # Создаем JWT токен
    access_token = create_access_token(data={"sub": user_id})

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            created_at=user_doc["created_at"],
            is_active=True
        )
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """Вход пользователя"""
    collection = get_collection("users")

    # Ищем пользователя по email или username
    user = await collection.find_one({
        "$or": [
            {"email": login_data.email_or_username},
            {"username": login_data.email_or_username}
        ]
    })

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Проверяем пароль
    if not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Проверяем, активен ли пользователь
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is disabled")

    # Создаем JWT токен
    access_token = create_access_token(data={"sub": str(user["_id"])})

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(user["_id"]),
            username=user["username"],
            email=user["email"],
            created_at=user["created_at"],
            is_active=user.get("is_active", True)
        )
    )


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    return current_user


@router.post("/auth/logout")
async def logout(current_user: UserResponse = Depends(get_current_user)):
    """Выход из системы (на клиенте нужно удалить токен)"""
    return {"message": "Successfully logged out"}


# ============ СУЩЕСТВУЮЩИЕ ЭНДПОИНТЫ (обновленные с аутентификацией) ============

@router.get("/health")
async def health_check():
    nn_ok = await nn_client.check_health()
    return {
        "status": "ok",
        "nn_service": "connected" if nn_ok else "unavailable",
    }


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
        body: SessionCreate,
        current_user: UserResponse = Depends(get_current_user)
):
    """Создание новой сессии чата для авторизованного пользователя"""
    system_prompt = await nn_client.get_system_prompt()

    session = {
        "user_id": current_user.id,
        "username": current_user.username,
        "created_at": datetime.utcnow(),
        "messages": [
            {"role": "system", "content": system_prompt, "timestamp": datetime.utcnow().isoformat()},
        ],
    }

    collection = get_collection("sessions")
    result = await collection.insert_one(session)

    # Добавляем ID сессии в список сессий пользователя
    users_collection = get_collection("users")
    await users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$push": {"sessions": str(result.inserted_id)}}
    )

    return SessionResponse(
        id=str(result.inserted_id),
        user_id=current_user.id,
        created_at=session["created_at"],
        message_count=0,
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(current_user: UserResponse = Depends(get_current_user)):
    """Получение списка сессий пользователя"""
    collection = get_collection("sessions")
    sessions = []
    async for doc in collection.find({"user_id": current_user.id}).sort("created_at", -1):
        user_messages = [m for m in doc["messages"] if m["role"] != "system"]
        sessions.append(SessionResponse(
            id=str(doc["_id"]),
            user_id=doc["user_id"],
            created_at=doc["created_at"],
            message_count=len(user_messages),
        ))
    return sessions


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, current_user: UserResponse = Depends(get_current_user)):
    collection = get_collection("sessions")
    try:
        doc = await collection.find_one({"_id": ObjectId(session_id), "user_id": current_user.id})
    except Exception as e:
        print(f"Error finding session: {e}")
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [m for m in doc["messages"] if m["role"] != "system"]
    print(f"Session {session_id}: total messages={len(doc['messages'])}, non-system={len(messages)}")
    return {"session_id": session_id, "messages": messages}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
        session_id: str,
        current_user: UserResponse = Depends(get_current_user)
):
    """Удаление сессии"""
    collection = get_collection("sessions")
    try:
        result = await collection.delete_one({"_id": ObjectId(session_id), "user_id": current_user.id})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
        session_id: str,
        body: ChatRequest,
        current_user: UserResponse = Depends(get_current_user)
):
    print(f"[CHAT] Session ID: {session_id}, User: {current_user.username}")
    collection = get_collection("sessions")

    # Поиск сессии
    try:
        doc = await collection.find_one({"_id": ObjectId(session_id), "user_id": current_user.id})
        print(f"[CHAT] Found session: {doc is not None}")
    except Exception as e:
        print(f"[CHAT] Error finding session: {e}")
        raise HTTPException(status_code=400, detail="Invalid session ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = {"role": "user", "content": body.message, "timestamp": datetime.utcnow().isoformat()}
    assistant_msg = {"role": "assistant", "content":  body.message, "timestamp": datetime.utcnow().isoformat()}

    # Обновление
    result = await collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$push": {"messages": {"$each": [user_msg, assistant_msg]}}}
    )
    print(f"[CHAT] Update result: matched={result.matched_count}, modified={result.modified_count}")

    # Проверка после обновления
    updated_doc = await collection.find_one({"_id": ObjectId(session_id)})
    print(f"[CHAT] Total messages now: {len(updated_doc.get('messages', []))}")

    return ChatResponse(response=body.message, session_id=session_id)


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket соединение для чата"""
    await websocket.accept()

    # Для WebSocket нужно передать токен в query параметре
    token = websocket.query_params.get("token")
    if not token:
        await websocket.send_json({"error": "No token provided"})
        await websocket.close()
        return

    # Верифицируем токен
    from auth import decode_access_token
    payload = decode_access_token(token)
    if not payload:
        await websocket.send_json({"error": "Invalid token"})
        await websocket.close()
        return

    user_id = payload.get("sub")
    if not user_id:
        await websocket.send_json({"error": "Invalid token"})
        await websocket.close()
        return

    collection = get_collection("sessions")
    try:
        doc = await collection.find_one({"_id": ObjectId(session_id), "user_id": user_id})
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
                response_text = f"Тестовый ответ: {body.message}"
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