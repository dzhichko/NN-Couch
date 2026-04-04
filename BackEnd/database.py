from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.DB_NAME]
        # Проверка подключения: попробуем выполнить простую команду
        await cls.client.admin.command('ping')
        print("Connected to MongoDB")

    @classmethod
    async def close(cls):
        if cls.client:
            cls.client.close()
            print("MongoDB connection closed")

    @classmethod
    def get_database(cls):
        """Возвращает объект базы данных"""
        return cls.db

# Вспомогательная функция для получения коллекции
def get_collection(name: str):
    return MongoDB.db[name]