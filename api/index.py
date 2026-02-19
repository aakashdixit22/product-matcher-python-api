from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Use sentence-transformers CLIP - better supported by HF Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")

if HF_TOKEN:
    logger.info(f"HF_TOKEN loaded: Yes (Token: {HF_TOKEN[:10]}...)")
else:
    logger.warning("HF_TOKEN not found — running unauthenticated (rate limits may apply)")


def generate_embedding(img):
    """Generate CLIP embedding via HF Inference API."""
    try:
        # Convert image to base64 (HF API accepts this format)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        logger.info(f"Image size: {len(img_bytes)} bytes")

        headers = {
            "Content-Type": "application/json"
        }
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # Send as JSON with base64-encoded image
        payload = {
            "inputs": img_base64
        }

        logger.info(f"Calling HF API: {HF_API_URL}")
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        logger.info(f"HF API status: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"HF API error {response.status_code}: {response.text}")

        result = response.json()
        logger.info(f"Raw response type: {type(result)}")

        # CLIP via HF Inference API returns a dict with 'image_embeds' key
        if isinstance(result, dict):
            if "image_embeds" in result:
                embedding = np.array(result["image_embeds"])
            elif "embeddings" in result:
                embedding = np.array(result["embeddings"])
            elif "data" in result:
                embedding = np.array(result["data"])
            else:
                # Last resort: flatten whatever values we got
                embedding = np.array(list(result.values())[0])
        elif isinstance(result, list):
            embedding = np.array(result)
        else:
            raise Exception(f"Unexpected response format: {type(result)}")

        # Flatten in case of extra dimensions (e.g. shape [1, 512])
        embedding = embedding.flatten()
        logger.info(f"Embedding shape after flatten: {embedding.shape}")

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    except Exception as e:
        logger.error(f"generate_embedding error: {e}", exc_info=True)
        raise


def decode_base64_image(b64_string):
    """Safely decode a base64 image string (with or without data URI prefix)."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "CLIP-ViT-B-32 (via HF API)",
        "deployment": "lightweight serverless",
        "version": "3.2",
        "authenticated": bool(HF_TOKEN),
    })


@app.route("/generate-embedding", methods=["POST"])
def generate_embedding_endpoint():
    try:
        logger.info("POST /generate-embedding")

        if "image" in request.files:
            logger.info("Source: file upload")
            file = request.files["image"]
            img = Image.open(file.stream).convert("RGB")
            source = "file"

        elif request.json and "imageUrl" in request.json:
            image_url = request.json["imageUrl"]
            logger.info(f"Source: URL — {image_url}")
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            source = "url"

        elif request.json and "imageBase64" in request.json:
            logger.info("Source: base64")
            img = decode_base64_image(request.json["imageBase64"])
            source = "base64"

        else:
            logger.warning("No image data in request")
            return jsonify({"error": "No image provided. Send 'image' file, 'imageUrl', or 'imageBase64'."}), 400

        embedding = generate_embedding(img)
        return jsonify({
            "embedding": embedding,
            "dimension": len(embedding),
            "source": source,
        })

    except Exception as e:
        logger.error(f"/generate-embedding error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/batch-embeddings", methods=["POST"])
def batch_embeddings():
    try:
        logger.info("POST /batch-embeddings")
        data = request.json or {}
        image_urls = data.get("imageUrls", [])

        if not image_urls:
            return jsonify({"error": "No image URLs provided"}), 400

        logger.info(f"Batch size: {len(image_urls)}")
        results = []

        for idx, url in enumerate(image_urls):
            try:
                logger.info(f"  [{idx+1}/{len(image_urls)}] {url}")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                embedding = generate_embedding(img)
                results.append({"url": url, "embedding": embedding, "success": True})
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({"url": url, "error": str(e), "success": False})

        successful = sum(1 for r in results if r["success"])
        logger.info(f"Batch done: {successful}/{len(image_urls)} successful")

        return jsonify({
            "results": results,
            "total": len(image_urls),
            "successful": successful,
        })

    except Exception as e:
        logger.error(f"/batch-embeddings error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/cosine-similarity", methods=["POST"])
def cosine_similarity():
    try:
        logger.info("POST /cosine-similarity")
        data = request.json or {}

        if "embedding1" not in data or "embedding2" not in data:
            return jsonify({"error": "Both 'embedding1' and 'embedding2' are required"}), 400

        e1 = np.array(data["embedding1"], dtype=np.float64)
        e2 = np.array(data["embedding2"], dtype=np.float64)

        if e1.shape != e2.shape:
            return jsonify({"error": f"Shape mismatch: {e1.shape} vs {e2.shape}"}), 400

        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return jsonify({"error": "One or both embeddings are zero vectors"}), 400

        similarity = float(np.dot(e1, e2) / (norm1 * norm2))
        logger.info(f"Cosine similarity: {similarity:.4f}")

        return jsonify({"similarity": similarity})

    except Exception as e:
        logger.error(f"/cosine-similarity error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)