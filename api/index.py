from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import os
import warnings
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel

load_dotenv()
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load CLIP model locally — no external API needed, 100% reliable
print("🔄 Loading CLIP model (first run downloads ~600MB)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model loaded!")

def generate_embedding(img):
    """Generate embedding using local CLIP model"""
    inputs = processor(images=img, return_tensors="pt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_features = model.get_image_features(**inputs)
    
    embedding = image_features.detach().numpy()[0]
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()


def decode_base64_image(b64_string):
    """Safely decode a base64 image string (with or without data URI prefix)."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'CLIP-ViT-B-32 (local)',
        'deployment': 'local model',
        'version': '4.0'
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
            img = decode_base64_image(request.json['imageBase64'])
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


# if __name__ == '__main__':
#     print("📍 Server running at http://localhost:5000")
#     app.run(host='0.0.0.0', port=5000, debug=True)