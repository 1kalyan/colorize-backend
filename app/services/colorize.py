# app/services/colorize.py

import os
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, LeakyReLU,
                                     MaxPooling2D, SeparableConv2D)
from tensorflow.keras.models import Model

from ..config import settings


# ---------- network (same as training) ----------
def _conv_block(x, filters: int):
    x = SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def _up_block(x, skip, filters: int):
    x = Conv2DTranspose(filters, 3, strides=2, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Concatenate()([x, skip])
    x = _conv_block(x, filters)
    return x

def build_generator(input_shape):
    inp = Input(shape=input_shape)
    c1 = _conv_block(inp, 32); p1 = MaxPooling2D()(c1)
    c2 = _conv_block(p1, 64);  p2 = MaxPooling2D()(c2)
    c3 = _conv_block(p2, 128); p3 = MaxPooling2D()(c3)
    c4 = _conv_block(p3, 256)
    u5 = _up_block(c4, c3, 128)
    u6 = _up_block(u5, c2, 64)
    u7 = _up_block(u6, c1, 32)
    out = Conv2D(3, 1, padding="same", activation="sigmoid", dtype="float32")(u7)
    return Model(inp, out, name="generator")


# ---------- model loader (full-model OR weights-only) ----------
_generator: Optional[Model] = None

def _try_load_full_model(path: str) -> Optional[Model]:
    """Return a compiled=False full model if `path` is a full Keras model, else None."""
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"[colorize] Loaded FULL model from {path}")
        return m
    except Exception as e:
        print(f"[colorize] Not a full model ({e}); will try weights-only…")
        return None

def _infer_hw_from_model(model: Model) -> Optional[tuple[int, int]]:
    try:
        # typical shape: (None, H, W, C)
        _, h, w, _ = model.input_shape
        if h and w:
            return int(h), int(w)
    except Exception:
        pass
    return None

def ensure_generator() -> Model:
    global _generator
    if _generator is not None:
        return _generator

    path = settings.MODEL_CHECKPOINT_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"MODEL_CHECKPOINT_PATH not found: {path}")

    # 1) try full model first
    model = _try_load_full_model(path)
    if model is None:
        # 2) fall back to weights-only
        size = int(settings.MODEL_INPUT_SIZE)
        model = build_generator((size, size, 1))
        model.load_weights(path)
        print(f"[colorize] Loaded WEIGHTS into fresh graph from {path}")

    _generator = model
    return _generator


# ---------- post-processing ----------
def sharpen_image_np(img: np.ndarray, amount: float = 0.5, radius: float = 1.2, threshold: float = 0.0) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_REFLECT)
    sharp = cv2.addWeighted(x, 1.0 + amount, blur, -amount, 0)
    if threshold > 0:
        mask = (np.abs(x - blur) < threshold)
        sharp = np.where(mask, x, sharp)
    return np.clip(sharp, 0.0, 1.0)

def fuse_luminance_with_pred(gray01: np.ndarray, pred_rgb01: np.ndarray) -> np.ndarray:
    g = gray01.squeeze().astype(np.float32)
    rgb = pred_rgb01.astype(np.float32)
    g8   = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lab  = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    lab[..., 0] = g8
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)

def clahe_on_l(rgb01: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    lab  = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[..., 0] = clahe.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)

def gray_world_wb(rgb01: np.ndarray) -> np.ndarray:
    # simple gray-world white balance on [0,1] float
    eps = 1e-6
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mr, mg, mb = r.mean() + eps, g.mean() + eps, b.mean() + eps
    gain_r, gain_b = mg / mr, mg / mb
    out = rgb01.copy()
    out[..., 0] *= gain_r
    out[..., 2] *= gain_b
    return np.clip(out, 0.0, 1.0)

def boost_saturation(rgb01: np.ndarray, factor: float = 1.05) -> np.ndarray:
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    hsv  = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    out8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out8.astype(np.float32) / 255.0


# ---------- inference ----------
def run_colorize(pil_image: Image.Image) -> Image.Image:
    """
    Accepts a PIL grayscale or RGB image.
    Returns a colorized PIL RGB image after fusion + CLAHE + sharpening.
    """
    gen = ensure_generator()

    # Use model-declared size if available; fallback to env MODEL_INPUT_SIZE
    hw = _infer_hw_from_model(gen)
    if hw is None:
        s = int(settings.MODEL_INPUT_SIZE)
        hw = (s, s)
    h, w = hw

    # 1) Build grayscale input that matches training
    gray = pil_image.convert("L") if pil_image.mode != "L" else pil_image
    gray = gray.resize((w, h), Image.BILINEAR)

    gray_arr = np.array(gray, dtype=np.float32) / 255.0
    gray_arr = gray_arr[..., None]  # (H, W, 1)
    model_in = np.expand_dims(gray_arr, axis=0)

    # 2) Run model
    pred = gen.predict(model_in, verbose=0)[0].astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)

    # 3) Post: fuse luminance + local contrast + mild sharpen
    fused = fuse_luminance_with_pred(gray_arr, pred)
    fused = clahe_on_l(fused, clip_limit=2.0, tile_grid_size=(8, 8))
    fused = sharpen_image_np(fused, amount=0.5, radius=1.2, threshold=0.01)
    fused = gray_world_wb(fused)
    fused = boost_saturation(fused, factor=1.05)

    out = (fused * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
