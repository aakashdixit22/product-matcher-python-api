from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import os

app = Flask(__name__)
CORS(app)

# Use Hugging Face Inference API (no local model needed!)
HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in Vercel environment variables

def generate_embedding(img):
    """Generate embedding using HF Inference API"""
    # Convert image to bytes
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    # Call HF API
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    response = requests.post(HF_API_URL, headers=headers, data=img_bytes)
    
    if response.status_code == 200:
        embedding = response.json()
        # Normalize
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    else:
        raise Exception(f"HF API error: {response.text}")

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
    try:
        if 'image' in request.files:
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
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'url'
            })
        elif request.json and 'imageBase64' in request.json:
            image_data = base64.b64decode(request.json['imageBase64'].split(',')[1])
            img = Image.open(BytesIO(image_data)).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'base64'
            })
        else:
            return jsonify({'error': 'No image provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-embeddings', methods=['POST'])
def batch_embeddings():
    try:
        data = request.json
        image_urls = data.get('imageUrls', [])
        
        if not image_urls:
            return jsonify({'error': 'No image URLs provided'}), 400
        
        results = []
        for idx, url in enumerate(image_urls):
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                embedding = generate_embedding(img)
                results.append({
                    'url': url,
                    'embedding': embedding,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'total': len(image_urls),
            'successful': sum(1 for r in results if r['success'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cosine-similarity', methods=['POST'])
def cosine_similarity():
    try:
        data = request.json
        embedding1 = np.array(data['embedding1'])
        embedding2 = np.array(data['embedding2'])
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return jsonify({'similarity': float(similarity)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
