# app/routes/auth.py
from fastapi import APIRouter, Depends, status
from pydantic import EmailStr

from ..deps import get_database  # -> returns your Motor/Mongo db
from ..schemas.user import TokenOut, UserLoginIn, UserOut, UserSignupIn
from ..services.auth_service import AuthService
from ..utils.security import create_access_token  # expects subject/user_id

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserSignupIn, db=Depends(get_database)):
    auth = AuthService(db)
    user = await auth.signup(payload)  # returns dict with id/email/full_name...
    token = create_access_token(user["id"])
    return TokenOut(access_token=token, user=UserOut(**user))

@router.post("/login", response_model=TokenOut)
async def login(payload: UserLoginIn, db=Depends(get_database)):
    auth = AuthService(db)
    user = await auth.login(payload.email, payload.password)
    token = create_access_token(user["id"])
    return TokenOut(access_token=token, user=UserOut(**user))
