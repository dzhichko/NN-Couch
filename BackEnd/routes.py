from fastapi import APIRouter, HTTPException, status
from models import ItemModel, ItemResponse
from database import get_collection
from bson import ObjectId
from typing import List

router = APIRouter()

# Health check
@router.get("/health")
async def health_check():
    """
    Проверка состояния сервера
    """
    # Можно добавить проверку соединения с БД, но для простоты возвращаем OK
    return {"status": "ok", "message": "Server is running"}

# Создание элемента (POST)
@router.post("/items", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemModel):
    """
    Добавление нового элемента в коллекцию items
    """
    collection = get_collection("items")
    # Преобразуем модель в словарь и вставляем
    item_dict = item.dict()
    result = await collection.insert_one(item_dict)
    # Получаем созданный документ
    created_item = await collection.find_one({"_id": result.inserted_id})
    return {
        "id": str(created_item["_id"]),
        "name": created_item["name"],
        "description": created_item.get("description"),
        "created_at": created_item["created_at"]
    }

# Получение всех элементов (GET)
@router.get("/items", response_model=List[ItemResponse])
async def get_items():
    """
    Возвращает список всех элементов из коллекции items
    """
    collection = get_collection("items")
    items = []
    async for doc in collection.find():
        items.append({
            "id": str(doc["_id"]),
            "name": doc["name"],
            "description": doc.get("description"),
            "created_at": doc["created_at"]
        })
    return items

# Опционально: эндпоинт для получения одного элемента по ID
@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str):
    """
    Получение элемента по его ObjectId
    """
    collection = get_collection("items")
    try:
        obj_id = ObjectId(item_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    doc = await collection.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Item not found")
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "description": doc.get("description"),
        "created_at": doc["created_at"]
    }