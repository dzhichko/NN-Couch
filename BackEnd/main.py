from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from database import MongoDB

app = FastAPI(title="NN-Couch Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    await MongoDB.connect()


@app.on_event("shutdown")
async def shutdown():
    await MongoDB.close()


@app.get("/")
async def root():
    return {"message": "NN-Couch Backend is running"}
