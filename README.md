# Image Caption API

A FastAPI service that generates descriptive captions for images using the `nlpconnect/vit-gpt2-image-captioning` model.

## Features

- Caption generation from image URLs
- LRU caching for improved performance
- Environment variable configuration
- Health check and monitoring endpoints
- Cache management utilities

## Requirements

- Python 3.12+
- Dependencies:
   ```
   fastapi
   uvicorn[standard]
   transformers
   torch
   Pillow
   accelerate
   aiohttp
   cachetools
   pydantic
   pydantic-settings
   python-dotenv
   sentencepiece
   hf_xet
   ```

## Setup

```bash
# Clone repository
git clone https://github.com/qwrurodhs/image-caption-api.git
cd image-caption-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Environment variables or `.env` file:

| Variable        | Description                    | Default                              |
|-----------------|--------------------------------|--------------------------------------|
| `LOG_LEVEL`     | Logging level                  | INFO                                 |
| `MODEL_NAME`    | Hugging Face model identifier  | nlpconnect/vit-gpt2-image-captioning |
| `CACHE_MAXSIZE` | Maximum items in LRU cache     | 512                                  |
| `HTTP_TIMEOUT`  | HTTP request timeout (seconds) | 3                                    |
| `DEVICE`        | Device for model execution     | cpu                                  |
| `PORT`          | Service port                   | 8000                                 |

## Usage

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Service available at http://localhost:8000

## API Endpoints

### Generate Captions

`POST /caption`

Request:

```json
{
  "image_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ]
}
```

Response:

```json
{
  "results": [
    {
      "url": "https://example.com/image1.jpg",
      "caption": "a cat sitting on a window sill",
      "processing_time": 0.456,
      "success": true,
      "cached": false
    },
    {
      "url": "https://example.com/image2.jpg",
      "caption": "a mountain landscape with snow capped peaks",
      "processing_time": 0.321,
      "success": true,
      "cached": false
    }
  ],
  "elapsed_time": 0.789
}
```

### Health Check

`GET /health`

### Cache Management

`GET /cache/info` - View cache status
`GET /cache/clear` - Clear cache

## Docker Deployment

```bash
docker build -t image-caption-api .
docker run -p 8000:8000 image-caption-api
```

## Performance Notes

- First request is slower due to model loading
- LRU cache improves performance for repeated requests
- CPU-optimized PyTorch used in Docker deployment

## Acknowledgements

Uses [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) model from
Hugging Face.