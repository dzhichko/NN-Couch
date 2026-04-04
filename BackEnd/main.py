from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from database import MongoDB

app = FastAPI(title="NN-Couch Backend", version="1.0.0")

# Настройка CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3243",
        "http://127.0.0.1:3243",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:63342",
        "http://127.0.0.1:63342",
        "http://localhost:63342/NN-Couch",  # Добавьте конкретный путь
        "*"  # Временно для теста
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.include_router(router)

@app.on_event("startup")
async def startup():
    await MongoDB.connect()
    # Создаем индексы для MongoDB
    db = MongoDB.get_database()
    await db.users.create_index("email", unique=True)
    await db.users.create_index("username", unique=True)
    await db.sessions.create_index("user_id")

@app.on_event("shutdown")
async def shutdown():
    await MongoDB.close()

@app.get("/")
async def root():
    return {"message": "NN-Couch Backend is running"}