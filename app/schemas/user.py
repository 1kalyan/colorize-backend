# app/schemas/user.py
from typing import Optional

from pydantic import (BaseModel, ConfigDict, EmailStr, Field, constr,
                      field_validator, model_validator)


# ---------- Requests ----------
class UserSignupIn(BaseModel):
    email: EmailStr
    password: constr(min_length=6, max_length=128) = Field(..., description="6–128 chars")
    full_name: Optional[constr(min_length=1, max_length=80)] = None
    first_name: Optional[constr(min_length=1, max_length=40)] = None
    last_name: Optional[constr(min_length=1, max_length=40)] = None

    model_config = ConfigDict(str_strip_whitespace=True, json_schema_extra={
        "examples": [{
            "email": "you@example.com",
            "password": "supersecret",
            "full_name": "John Doe"
        }]
    })

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        return str(v).lower()

    @model_validator(mode="after")
    def ensure_name(self):
        if not self.full_name and not (self.first_name and self.last_name):
            raise ValueError("Provide either full_name or both first_name and last_name.")
        if not self.full_name and self.first_name and self.last_name:
            self.full_name = f"{self.first_name} {self.last_name}".strip()
        if self.full_name:
            self.full_name = " ".join(self.full_name.split())
        return self


class UserLoginIn(BaseModel):
    email: EmailStr
    password: constr(min_length=1)

# ---------- Responses ----------
class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    cloudinary_public_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
