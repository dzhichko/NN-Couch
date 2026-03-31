from fastapi import FastAPI
from routes import router
from database import MongoDB

app = FastAPI(title="Chat Backend", version="1.0.0")

# Подключаем роутер
app.include_router(router)

# События при старте и остановке приложения
@app.on_event("startup")
async def startup():
    await MongoDB.connect()

@app.on_event("shutdown")
async def shutdown():
    await MongoDB.close()

# Корневой эндпоинт (опционально)
@app.get("/")
async def root():
    return {"message": "Chat Backend is running"}