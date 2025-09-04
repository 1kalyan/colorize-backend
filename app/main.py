from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import auth as auth_router
from .routers import images as images_router

app = FastAPI(title=settings.APP_NAME, debug=settings.APP_DEBUG)

app.include_router(auth_router.router)
app.include_router(images_router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://1kalyan.github.io",   # <— add this (no /frontend)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["meta"])
def root():
    return {"ok": True, "app": settings.APP_NAME}
