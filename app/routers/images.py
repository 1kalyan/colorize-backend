# app/routers/images.py
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId
from fastapi import (APIRouter, Depends, File, Header, HTTPException, Query,
                     Request, UploadFile)
from PIL import Image
from starlette.concurrency import run_in_threadpool

from ..deps import bearer_token, get_current_user_claims, get_database
from ..schemas.image import ImageRecordOut
from ..services.cloudinary_service import build_signed_url, upload_image
from ..services.colorize import run_colorize

router = APIRouter(prefix="/images", tags=["images"])

def oid_str(x) -> str: return str(x) if isinstance(x, ObjectId) else x


@router.post("/colorize", response_model=ImageRecordOut)
async def upload_and_colorize(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    db = Depends(get_database),
):
    token = bearer_token(authorization)
    claims = get_current_user_claims(token)
    user_id = claims["sub"]

    # Save upload to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        temp_in = tmp.name
        content = await file.read()
        tmp.write(content)

    # Upload ORIGINAL to Cloudinary (authenticated)
    orig_pub_prefix = f"{user_id}/original/{uuid4()}"
    orig_info = upload_image(temp_in, public_id_prefix=orig_pub_prefix)
    original_public_id = orig_info["public_id"]

    # Colorize (in a worker thread)
    def _do_colorize():
        img = Image.open(temp_in)
        out = run_colorize(img)  # your function returns a PIL.Image
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        out.save(temp_out, format="PNG")
        return temp_out

    temp_out = await run_in_threadpool(_do_colorize)

    # Upload COLORIZED
    col_pub_prefix = f"{user_id}/colorized/{uuid4()}"
    col_info = upload_image(temp_out, public_id_prefix=col_pub_prefix)
    colorized_public_id = col_info["public_id"]

    created_at = datetime.now(timezone.utc)

    # Store record in Mongo (store PUBLIC IDs; do not store public URLs)
    doc = {
        "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
        "original_public_id": original_public_id,
        "colorized_public_id": colorized_public_id,
        "created_at": created_at,
    }
    res = await db.images.insert_one(doc)

    # Cleanup temps
    for p in (temp_in, temp_out):
        try:
            os.remove(p)
        except:
            pass

    # Mint short-lived signed URLs for immediate use by the frontend
    original_url = build_signed_url(original_public_id, expires_in=15 * 60)
    colorized_url = build_signed_url(colorized_public_id, expires_in=15 * 60)

    return ImageRecordOut(
        id=str(res.inserted_id),
        original_url=original_url,
        colorized_url=colorized_url,
        output_url=colorized_url,   # alias for your Colorize page
        created_at=created_at,
    )


@router.get("/", response_model=list[ImageRecordOut])
async def list_my_images(
    authorization: str | None = Header(default=None),
    db = Depends(get_database),
    limit: int = Query(60, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    token = bearer_token(authorization)
    claims = get_current_user_claims(token)
    user_id = claims["sub"]

    q = {"user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id}

    items: list[ImageRecordOut] = []
    cursor = db.images.find(q).sort("created_at", -1).skip(offset).limit(limit)
    async for x in cursor:
        # ---- Backward compatibility & safety ----
        # original
        orig_url = None
        if x.get("original_public_id"):
            orig_url = build_signed_url(x["original_public_id"], expires_in=15 * 60)
        elif x.get("original_url"):
            # legacy docs stored a direct URL
            orig_url = x["original_url"]

        if not orig_url:
            # bad/partial row -> skip instead of 500
            continue

        # colorized (optional)
        col_url = None
        if x.get("colorized_public_id"):
            col_url = build_signed_url(x["colorized_public_id"], expires_in=15 * 60)
        elif x.get("colorized_url"):
            col_url = x["colorized_url"]

        created_at = x.get("created_at") or datetime.now(timezone.utc)

        items.append(ImageRecordOut(
            id=oid_str(x["_id"]),
            original_url=orig_url,
            colorized_url=col_url,
            output_url=col_url,
            created_at=created_at,
        ))

    return items
