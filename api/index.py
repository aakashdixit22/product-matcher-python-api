from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Load MobileNetV2 model (pre-trained on ImageNet)
# Remove the top classification layer to get embeddings
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("✅ MobileNetV2 model loaded successfully")

def process_image(img):
    """Process image to the format required by MobileNetV2"""
    # Resize to 224x224
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for MobileNetV2
    img_array = preprocess_input(img_array)
    return img_array

def generate_embedding(img):
    """Generate embedding vector from image"""
    processed_img = process_image(img)
    embedding = model.predict(processed_img, verbose=0)
    # Flatten to 1D array
    return embedding.flatten().tolist()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'MobileNetV2',
        'version': '1.0'
    })

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding_endpoint():
    """Generate embedding from uploaded image or URL"""
    try:
        # Check if image file was uploaded
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB')
            embedding = generate_embedding(img)
            return jsonify({
                'embedding': embedding,
                'dimension': len(embedding),
                'source': 'file'
            })
        
        # Check if image URL was provided
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
        
        # Check if base64 image was provided
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
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-embeddings', methods=['POST'])
def batch_embeddings():
    """Generate embeddings for multiple image URLs"""
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
                print(f"✅ Processed {idx + 1}/{len(image_urls)}")
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'success': False
                })
                print(f"❌ Failed {idx + 1}/{len(image_urls)}: {str(e)}")
        
        return jsonify({
            'results': results,
            'total': len(image_urls),
            'successful': sum(1 for r in results if r['success'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cosine-similarity', methods=['POST'])
def cosine_similarity():
    """Calculate cosine similarity between two embeddings"""
    try:
        data = request.json
        embedding1 = np.array(data['embedding1'])
        embedding2 = np.array(data['embedding2'])
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return jsonify({
            'similarity': float(similarity)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
