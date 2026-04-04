import os


class Settings:
    # Используем аутентификацию с root пользователем
    # Формат: mongodb://username:password@localhost:27017/dbname?authSource=admin
    MONGODB_URL: str = os.getenv(
        "MONGODB_URL",
        "mongodb://admin:secret@localhost:27017/chat_db?authSource=admin"
    )
    DB_NAME: str = os.getenv("DB_NAME", "chat_db")
    NN_SERVICE_URL: str = os.getenv("NN_SERVICE_URL", "http://localhost:8001")

    # JWT настройки
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


settings = Settings()
