from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    APP_NAME: str = "ColorizeAPI"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days

    MONGO_URL: str
    MONGO_DB: str = "colorize_db"

    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    CLOUDINARY_FOLDER: str = "colorize_app"

    MODEL_CHECKPOINT_PATH: str = "/app/artifacts/best_generator.weights.h5"
    MODEL_INPUT_SIZE: int = Field(default=256, ge=64, le=1024)

settings = Settings()
