import asyncio
import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional, Any, Annotated

import aiohttp
from cachetools import LRUCache
from fastapi import FastAPI, HTTPException, Request, Depends
from PIL import Image
from pydantic import AnyHttpUrl, BaseModel
from pydantic_settings import BaseSettings
from transformers import pipeline


class AppSettings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    MODEL_NAME: str = "nlpconnect/vit-gpt2-image-captioning"
    CACHE_MAXSIZE: int = 512
    HTTP_TIMEOUT: int = 10
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    MAX_URLS_PER_REQUEST: int = 20
    DEVICE: str = "cpu"
    PORT: int = 8000
    HTTP_USER_AGENT: str = "ImageCaptionBot/1.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = AppSettings()

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class EnhancedCache(LRUCache):
    def __init__(self, maxsize: int):
        super().__init__(maxsize)
        self.hits = 0
        self.misses = 0

    def __getitem__(self, key: Any) -> Any:
        try:
            value = super().__getitem__(key)
            self.hits += 1
            return value
        except KeyError:
            self.misses += 1
            raise

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class CaptionRequest(BaseModel):
    image_urls: List[AnyHttpUrl]


class ImageResult(BaseModel):
    url: str
    caption: Optional[str] = None
    processing_time: float
    success: bool
    cached: bool = False


class CaptionResponse(BaseModel):
    results: List[ImageResult]


async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    try:
        headers = {"User-Agent": settings.HTTP_USER_AGENT}
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > settings.MAX_IMAGE_SIZE:
                logger.error(f"Image too large: {url} ({content_length} bytes)")
                return None

            return await response.read()

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error fetching {url}: {type(e).__name__} - {e}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching {url}")
    return None


async def generate_caption(image_pipeline, image_data: bytes) -> Optional[str]:
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        result = await asyncio.to_thread(
            image_pipeline,
            image,
            max_new_tokens=100
        )
        caption = result[0].get("generated_text", "").strip()
        return caption or None

    except Exception as e:
        logger.error(f"Caption generation error: {type(e).__name__} - {e}")
        return None


async def process_image(session: aiohttp.ClientSession, url: str, image_pipeline, cache: EnhancedCache) -> ImageResult:
    start_time = time.monotonic()
    cache_key = str(url)

    if cache_key in cache:
        cached_data = cache[cache_key]
        new_processing_time = time.monotonic() - start_time
        logger.debug(f"Cache hit for {url}")

        response_data = cached_data.copy()
        response_data['processing_time'] = new_processing_time
        response_data['cached'] = True

        return ImageResult(**response_data)

    image_data = await fetch_image(session, url)

    if image_data:
        caption = await generate_caption(image_pipeline, image_data)
        success = caption is not None
    else:
        caption = "Failed to fetch image."
        success = False

    processing_time = time.monotonic() - start_time
    result_dict = {
        "url": url,
        "caption": caption,
        "success": success,
        "processing_time": processing_time,
    }

    if success:
        cache[cache_key] = result_dict

    logger.info(f"Processed {url} in {processing_time:.2f}s - Success: {success}")
    return ImageResult(**result_dict)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting Image Caption API...")

    try:
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        application.state.image_pipeline = pipeline(
            "image-to-text", model=settings.MODEL_NAME, device=settings.DEVICE
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}", exc_info=True)
        raise RuntimeError("Model initialization failed") from e

    application.state.url_cache = EnhancedCache(maxsize=settings.CACHE_MAXSIZE)
    application.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=settings.HTTP_TIMEOUT)
    )
    logger.info("HTTP client and cache initialized.")

    logger.info("Application startup complete.")
    yield

    logger.info("Shutting down application...")
    await application.state.http_session.close()
    logger.info("HTTP client closed.")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Image Caption API",
    description="AI-powered image captioning service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)


def get_http_session(request: Request) -> aiohttp.ClientSession:
    return request.app.state.http_session


def get_image_pipeline(request: Request):
    return request.app.state.image_pipeline


def get_url_cache(request: Request) -> EnhancedCache:
    return request.app.state.url_cache


@app.post("/caption", response_model=CaptionResponse, summary="Generate image captions")
async def caption_images(
        payload: CaptionRequest,
        session: Annotated[aiohttp.ClientSession, Depends(get_http_session)],
        image_pipeline: Annotated[Any, Depends(get_image_pipeline)],
        cache: Annotated[EnhancedCache, Depends(get_url_cache)],
):
    if len(payload.image_urls) > settings.MAX_URLS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_URLS_PER_REQUEST} URLs per request allowed."
        )

    tasks = [process_image(session, str(url), image_pipeline, cache) for url in payload.image_urls]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.success)
    logger.info(f"Processed {len(results)} images - {successful} successful")

    return CaptionResponse(results=results)


@app.get("/health", summary="Health check")
async def health_check(request: Request, pipeline: Annotated[Any, Depends(get_image_pipeline)]):
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "cache_size": len(request.app.state.url_cache),
        "timestamp": time.time()
    }


@app.get("/cache/info", summary="Cache information")
async def cache_info(cache: Annotated[EnhancedCache, Depends(get_url_cache)]):
    return {
        "current_size": len(cache),
        "max_size": cache.maxsize,
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": f"{cache.hit_rate:.2%}"
    }


@app.delete("/cache/clear", summary="Clear cache")
async def clear_cache(cache: Annotated[EnhancedCache, Depends(get_url_cache)]):
    previous_size = len(cache)
    cache.clear()
    cache.hits = 0
    cache.misses = 0
    logger.info(f"Cache cleared - {previous_size} items removed")
    return {"message": "Cache cleared successfully", "previous_size": previous_size}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on http://0.0.0.0:{settings.PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
