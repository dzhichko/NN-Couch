import os


class Settings:
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "chat_db")
    NN_SERVICE_URL: str = os.getenv("NN_SERVICE_URL", "http://localhost:8001")


settings = Settings()
