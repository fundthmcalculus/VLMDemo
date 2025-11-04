import base64
import io
import os
import time
import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, request, render_template
from PIL import Image

from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

# Choose a small, CPU-friendly image captioning model by default.
# You can override via env var CAPTION_MODEL if desired.
MODEL_NAME = os.environ.get("CAPTION_MODEL", "nlpconnect/vit-gpt2-image-captioning")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("vlm_demo")

# ---------- Text-to-Image (T2I) configuration ----------
T2I_DEFAULT_MODEL = os.environ.get("T2I_MODEL", "runwayml/stable-diffusion-v1-5")
T2I_DEFAULT_MODELS = [
    "amd/Nitro-E",                    # Requested default
    "stabilityai/sd-turbo",          # Fast SD on small GPUs/CPUs
    "runwayml/stable-diffusion-v1-5",# Widely available baseline
    "Lykon/dreamshaper-6",           # Popular SD 1.5 derivative
]
ALLOWED_T2I_MODELS = [m.strip() for m in os.environ.get("ALLOWED_T2I_MODELS", ",".join(T2I_DEFAULT_MODELS)).split(",") if m.strip()]

# Caches for pipelines
_t2i_cache = {}
_device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu")
_dtype = torch.float16 if _device in ("cuda", "mps") else torch.float32

# Curated small vision captioning models (mostly CPU-friendly). Override with ALLOWED_MODELS env (comma-separated) if desired.
DEFAULT_MODELS = [
    "nlpconnect/vit-gpt2-image-captioning",           # ViT-GPT2 captioning (default)
    "Salesforce/blip-image-captioning-base",          # BLIP base captioning
    "microsoft/git-base",                             # GIT base (captioning capable)
    "ydshieh/vit-gpt2-coco-en",                       # ViT-GPT2 trained on COCO
]
ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]

# Pipeline cache and lock for thread safety
_captioner = None
_pipeline_cache = {}
_cache_lock = threading.Lock()


def _build_pipeline(model_name: str):
    logger.info(f"Loading caption model: {model_name}")
    t0 = time.time()
    pipe = pipeline("image-to-text", model=model_name)
    logger.info(f"Model {model_name} loaded in {time.time() - t0:.2f}s")
    return pipe


def init_captioner():
    global _captioner
    if _captioner is None:
        _captioner = _build_pipeline(MODEL_NAME)
        with _cache_lock:
            _pipeline_cache[MODEL_NAME] = _captioner


def get_pipeline_for_model(model_name: str):
    # Enforce whitelist to avoid arbitrary downloads unless explicitly allowed
    if model_name not in ALLOWED_MODELS:
        logger.warning(f"Requested model '{model_name}' not in allowed list; using default '{MODEL_NAME}'")
        model_name = MODEL_NAME

    with _cache_lock:
        if model_name in _pipeline_cache:
            logger.debug(f"Using cached pipeline for {model_name}")
            return _pipeline_cache[model_name]

    # Build outside lock to avoid blocking other requests during download/init
    pipe = _build_pipeline(model_name)
    with _cache_lock:
        _pipeline_cache[model_name] = pipe
    return pipe


def get_captioner():
    # Backward-compatible accessor for default model
    return get_pipeline_for_model(MODEL_NAME)

# ----------------- Text-to-Image helpers -----------------

def _build_t2i_pipeline(model_name: str):
    logger.info(f"Loading T2I model: {model_name} on device={_device} dtype={_dtype}")
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=_dtype)
    try:
        pipe.to(_device)
    except Exception:
        logger.exception("Failed to move pipeline to device, falling back to CPU")
        pipe.to("cpu")
    # Some memory optimizations for low VRAM/CPU
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    load_s = time.time() - t0
    logger.info(f"T2I model {model_name} loaded in {load_s:.2f}s")
    return pipe


def get_t2i_pipeline_for_model(model_name: str):
    if model_name not in ALLOWED_T2I_MODELS:
        logger.warning(f"Requested T2I model '{model_name}' not allowed; using default '{T2I_DEFAULT_MODEL}'")
        model_name = T2I_DEFAULT_MODEL
    with _cache_lock:
        if model_name in _t2i_cache:
            logger.debug(f"Using cached T2I pipeline for {model_name}")
            return _t2i_cache[model_name]
    pipe = _build_t2i_pipeline(model_name)
    with _cache_lock:
        _t2i_cache[model_name] = pipe
    return pipe


def init_t2i():
    try:
        get_t2i_pipeline_for_model(T2I_DEFAULT_MODEL)
        logger.info(f"Preloaded T2I default model: {T2I_DEFAULT_MODEL}")
    except Exception:
        logger.exception("Failed to preload T2I model")


@app.get("/")
def index():
    # Render the extracted template
    return render_template("index.html")


@app.get("/text2image")
def text2image_page():
    return render_template("text2image.html")


@app.get("/api/models")
def list_models():
    """Return allowed caption models and flag the default."""
    models = []
    with _cache_lock:
        loaded = set(_pipeline_cache.keys())
    for m in ALLOWED_MODELS:
        models.append({
            "id": m,
            "display": m.split("/")[-1],
            "default": m == MODEL_NAME,
            "loaded": m in loaded,
        })
    logger.debug("Exposing caption model list: %s", models)
    return jsonify({
        "models": models,
        "default": MODEL_NAME,
    })


@app.get("/api/t2i_models")
def list_t2i_models():
    """Return allowed text-to-image models and flag the default."""
    models = []
    with _cache_lock:
        loaded = set(_t2i_cache.keys())
    for m in ALLOWED_T2I_MODELS:
        models.append({
            "id": m,
            "display": m.split("/")[-1],
            "default": m == T2I_DEFAULT_MODEL,
            "loaded": m in loaded,
        })
    logger.debug("Exposing T2I model list: %s", models)
    return jsonify({
        "models": models,
        "default": T2I_DEFAULT_MODEL,
    })


def _img_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/text2image")
def text2image():
    req_id = datetime.utcnow().strftime("%H%M%S%f")
    t0 = time.time()
    try:
        data = request.get_json(silent=True) or {}
        # Allow form too
        if not data and request.form:
            data = request.form
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            logger.warning(f"{req_id} no prompt provided from {request.remote_addr}")
            return jsonify({"error": "No prompt provided"}), 400
        selected_model = data.get("model") or T2I_DEFAULT_MODEL
        steps = int(data.get("steps") or os.environ.get("T2I_STEPS", "6"))
        guidance = float(data.get("guidance_scale") or os.environ.get("T2I_GUIDANCE", "0.0"))
        width = int(data.get("width") or 512)
        height = int(data.get("height") or 512)

        logger.info(f"{req_id} t2i prompt_len={len(prompt)} model={selected_model} from {request.remote_addr} ua={request.user_agent.string}")

        pipe = get_t2i_pipeline_for_model(selected_model)
        t_inf0 = time.time()
        with torch.inference_mode():
            images = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, width=width, height=height).images
        infer_ms = int((time.time() - t_inf0) * 1000)
        img = images[0]
        b64 = _img_to_base64_jpeg(img)
        total_ms = int((time.time() - t0) * 1000)
        logger.info(f"{req_id} t2i ok model={selected_model} infer={infer_ms}ms total={total_ms}ms size={width}x{height}")
        return jsonify({
            "image_base64": f"data:image/jpeg;base64,{b64}",
            "model": selected_model,
            "infer_ms": infer_ms,
            "total_ms": total_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    except Exception:
        logger.exception(f"{req_id} t2i error processing prompt from {request.remote_addr}")
        return jsonify({"error": "internal server error"}), 500


@app.post("/api/describe")
def describe_image():
    req_id = datetime.utcnow().strftime("%H%M%S%f")
    t0 = time.time()
    try:
        # Accept either multipart/form-data under key 'image' or JSON with 'image_base64'
        selected_model = None
        if "image" in request.files:
            file = request.files["image"]
            img_bytes = file.read()
            source = f"multipart:{getattr(file, 'filename', 'frame.jpg')}"
            # Optional model selection in multipart form
            selected_model = request.form.get("model")
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_base64", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64) if b64 else None
            source = "json:base64"
            selected_model = data.get("model")

        if not img_bytes:
            logger.warning(f"{req_id} no image provided from {request.remote_addr}")
            return jsonify({"error": "No image provided"}), 400

        model_to_use = selected_model or MODEL_NAME
        logger.info(
            f"{req_id} recv image bytes={len(img_bytes)} model={model_to_use} from {request.remote_addr} ua={request.user_agent.string}"
        )

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        cap = get_pipeline_for_model(model_to_use)
        t_inf0 = time.time()
        results = cap(image)
        infer_ms = int((time.time() - t_inf0) * 1000)
        # Handle common return schema variations
        text = None
        if isinstance(results, list) and results:
            item = results[0]
            text = item.get("generated_text") or item.get("caption")
        if not text:
            text = "(no caption)"

        total_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"{req_id} ok model={getattr(cap, 'model', None)} infer={infer_ms}ms total={total_ms}ms src={source} text_len={len(text)}"
        )
        return jsonify({
            "description": text,
            "model": model_to_use,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    except Exception:
        logger.exception(f"{req_id} error processing image from {request.remote_addr}")
        return jsonify({"error": "internal server error"}), 500


@app.get("/health")
def health():
    with _cache_lock:
        loaded_caps = list(_pipeline_cache.keys())
        loaded_t2i = list(_t2i_cache.keys())
    return jsonify({
        "status": "ok",
        "caption": {
            "default_model": MODEL_NAME,
            "loaded": _captioner is not None,
            "allowed_models": ALLOWED_MODELS,
            "loaded_models": loaded_caps,
        },
        "t2i": {
            "default_model": T2I_DEFAULT_MODEL,
            "allowed_models": ALLOWED_T2I_MODELS,
            "loaded_models": loaded_t2i,
            "device": _device,
            "dtype": str(_dtype),
        }
    })


if __name__ == '__main__':
    # Eagerly load the default model at startup
    init_captioner()
    # Preload the default T2I model in a background thread
    init_t2i()
    # Bind to all interfaces so it can be opened from other devices if needed
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
