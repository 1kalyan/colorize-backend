# app/services/auth_service.py
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from fastapi import HTTPException

from ..schemas.user import UserSignupIn
from ..utils.security import hash_password, verify_password


def _str_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Convert Mongo _id -> id:str
    if doc and "_id" in doc:
        doc = dict(doc)
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

class AuthService:
    def __init__(self, db):
        self.db = db

    async def _ensure_unique_email_index(self):
        # Safe to call often; MongoDB keeps one unique index
        await self.db.users.create_index("email", unique=True)

    async def signup(self, payload: UserSignupIn) -> Dict[str, Any]:
        await self._ensure_unique_email_index()
        email = str(payload.email).lower()

        if await self.db.users.find_one({"email": email}):
            raise HTTPException(status_code=400, detail="Email already exists")

        doc = {
            "email": email,
            "password": hash_password(payload.password),
            "full_name": payload.full_name,
            "avatar_url": None,
            "cloudinary_public_id": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        res = await self.db.users.insert_one(doc)
        created = await self.db.users.find_one({"_id": res.inserted_id})
        return _str_id(created)

    async def login(self, email: str, password: str) -> Dict[str, Any]:
        email = email.lower()
        user = await self.db.users.find_one({"email": email})
        if not user or not verify_password(password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return _str_id(user)
