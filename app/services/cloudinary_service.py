import time

import cloudinary
import cloudinary.uploader as uploader
from cloudinary.utils import cloudinary_url

from ..config import settings


def init_cloudinary():
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True,
    )

def upload_image(file_path: str, public_id_prefix: str) -> dict:
    """
    Uploads a local file to Cloudinary as an AUTHENTICATED asset.
    Returns dict with public_id and (internal) secure_url.
    """
    init_cloudinary()
    result = uploader.upload(
        file_path,
        folder=settings.CLOUDINARY_FOLDER,   # e.g. "colorization"
        public_id=public_id_prefix,          # e.g. "<userId>/original/<uuid>"
        overwrite=True,
        resource_type="image",
        type="authenticated",                # <-- key: make it private/authenticated
    )
    return {"public_id": result["public_id"], "secure_url": result["secure_url"]}

def build_signed_url(public_id: str, expires_in: int = 15 * 60) -> str:
    """
    Builds a time-limited signed URL for viewing/downloading an authenticated asset.
    """
    if not public_id:
        return ""
    init_cloudinary()
    expires_at = int(time.time()) + int(expires_in)
    # Returns (url, options); take url[0]
    url, _opts = cloudinary_url(
        public_id,
        type="authenticated",
        sign_url=True,
        expires_at=expires_at,
        secure=True,
    )
    return url
