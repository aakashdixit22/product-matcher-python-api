# Python AI API for Visual Product Matcher

This is the AI/ML service that handles image embeddings using TensorFlow and MobileNetV2.

## Setup

1. **Create a virtual environment** (recommended):
```bash
cd python-api
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the API server**:
```bash
python app.py
```

The server will start at `http://localhost:5000`

## API Endpoints

### GET /health
Health check endpoint
```bash
curl http://localhost:5000/health
```

### POST /generate-embedding
Generate embedding from image file or URL

**With file upload:**
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/generate-embedding
```

**With image URL:**
```bash
curl -X POST http://localhost:5000/generate-embedding \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/image.jpg"}'
```

**Response:**
```json
{
  "embedding": [0.123, 0.456, ...],
  "dimension": 1280,
  "source": "file"
}
```

### POST /batch-embeddings
Generate embeddings for multiple image URLs

```bash
curl -X POST http://localhost:5000/batch-embeddings \
  -H "Content-Type: application/json" \
  -d '{"imageUrls": ["url1", "url2", ...]}'
```

### POST /cosine-similarity
Calculate similarity between two embeddings

```bash
curl -X POST http://localhost:5000/cosine-similarity \
  -H "Content-Type: application/json" \
  -d '{"embedding1": [...], "embedding2": [...]}'
```

## Model Details

- **Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: 1280-dimensional embedding vector
- **Pooling**: Global Average Pooling

## Notes

- The model is loaded once on startup and kept in memory
- All images are automatically resized to 224x224
- CORS is enabled for Next.js frontend integration
- The API runs on port 5000 by default

