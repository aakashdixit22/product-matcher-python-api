


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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Use Hugging Face Inference API (no local model needed!)
HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
# Try both possible environment variable names
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")


if HF_TOKEN:
    logger.info(f"HF_TOKEN loaded: Yes (Token: {HF_TOKEN[:10]}...)")
else:
    logger.info("HF_TOKEN loaded: No (running without auth)")

def generate_embedding(img):
    """Generate embedding using HF Inference API"""
    try:
        # Convert image to bytes
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        logger.info(f"Image converted to bytes: {len(img_bytes)} bytes")
        
        # Call HF API
        headers = {"Content-Type": "application/octet-stream"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
            logger.info("Using authenticated HF API request")
        else:
            logger.warning("No HF_TOKEN found, using unauthenticated request (may have rate limits)")
        
        logger.info(f"Calling HF API: {HF_API_URL}")
        response = requests.post(HF_API_URL, headers=headers, data=img_bytes)
        logger.info(f"HF API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            
            # HF Inference API returns embeddings in different formats depending on the model
            # For CLIP, it might be a direct array or nested in a key
            if isinstance(result, list):
                embedding = np.array(result)
            elif isinstance(result, dict):
                # Try common keys where embeddings might be stored
                embedding = np.array(result.get('embeddings', result.get('data', result)))
            else:
                embedding = np.array(result)
            
            logger.info(f"Embedding shape: {embedding.shape}")
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            logger.info("Embedding generated successfully")
            return embedding.tolist()
        else:
            error_msg = f"HF API error (status {response.status_code}): {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
    except Exception as e:
        logger.error(f"Error in generate_embedding: {str(e)}", exc_info=True)
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'CLIP-ViT-B-32 (via HF API)',
        'deployment': 'lightweight serverless',
        'version': '3.1'
    })

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding_endpoint():
    print(HF_TOKEN)  # Debug: Print the token to verify it's loaded (remove in production)
    try:
        logger.info("Received request to /generate-embedding")
        
        if 'image' in request.files:
            logger.info("Processing image from file upload")
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'file'
            })
        elif request.json and 'imageUrl' in request.json:
            image_url = request.json['imageUrl']
            logger.info(f"Processing image from URL: {image_url}")
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'url'
            })
        elif request.json and 'imageBase64' in request.json:
            logger.info("Processing image from base64")
            image_data = base64.b64decode(request.json['imageBase64'].split(',')[1])
            img = Image.open(BytesIO(image_data)).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'base64'
            })
        else:
            logger.warning("No image data found in request")
            return jsonify({'error': 'No image provided'}), 400
    except Exception as e:
        logger.error(f"Error in /generate-embedding endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/batch-embeddings', methods=['POST'])
def batch_embeddings():
    try:
        logger.info("Received request to /batch-embeddings")
        data = request.json
        image_urls = data.get('imageUrls', [])
        
        if not image_urls:
            logger.warning("No image URLs provided in batch request")
            return jsonify({'error': 'No image URLs provided'}), 400
        
        logger.info(f"Processing batch of {len(image_urls)} images")
        results = []
        for idx, url in enumerate(image_urls):
            try:
                logger.info(f"Processing image {idx+1}/{len(image_urls)}: {url}")
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                embedding = generate_embedding(img)
                results.append({
                    'url': url,
                    'embedding': embedding,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Error processing image {idx+1}: {str(e)}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'success': False
                })
        
        logger.info(f"Batch complete: {sum(1 for r in results if r['success'])}/{len(image_urls)} successful")
        return jsonify({
            'results': results,
            'total': len(image_urls),
            'successful': sum(1 for r in results if r['success'])
        })
    except Exception as e:
        logger.error(f"Error in /batch-embeddings endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/cosine-similarity', methods=['POST'])
def cosine_similarity():
    try:
        logger.info("Received request to /cosine-similarity")
        data = request.json
        embedding1 = np.array(data['embedding1'])
        embedding2 = np.array(data['embedding2'])
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        logger.info(f"Cosine similarity calculated: {similarity}")
        
        return jsonify({'similarity': float(similarity)})
    except Exception as e:
        logger.error(f"Error in /cosine-similarity endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
