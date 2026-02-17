# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import requests
# from io import BytesIO
# from PIL import Image
# import base64
# from transformers import CLIPProcessor, CLIPModel
# import warnings
# warnings.filterwarnings('ignore')

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Next.js frontend

# print("🔄 Loading CLIP model (this may take a minute on first run)...")
# # Use CLIP model - much better semantic understanding
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# print("✅ CLIP model loaded! Semantic understanding enabled")
# print("📦 Size: ~150MB (vs 605MB PyTorch full install)")
# print("🎯 Can now distinguish: cars vs bags vs shoes vs furniture etc.")


# def generate_embedding(img):
#     """Generate semantic embedding using CLIP model"""
#     # Process image with CLIP
#     inputs = processor(images=img, return_tensors="pt")
    
#     # Generate image features
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         image_features = model.get_image_features(**inputs)
    
#     # Convert to numpy and normalize
#     embedding = image_features.detach().numpy()[0]
    
#     # L2 normalization for cosine similarity
#     norm = np.linalg.norm(embedding)
#     if norm > 0:
#         embedding = embedding / norm
    
#     return embedding.tolist()

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model': 'CLIP-ViT-B-32 (OpenAI)',
#         'understanding': 'semantic (can distinguish object categories)',
#         'version': '3.0',
#         'deployment': 'optimized (~150MB vs 605MB full PyTorch)',
#         'embedding_size': 512,
#         'performance': 'High accuracy for visual similarity',
#         'advantages': 'Much better semantic understanding (cars vs bags)'
#     })

# @app.route('/generate-embedding', methods=['POST'])
# def generate_embedding_endpoint():
#     """Generate embedding from uploaded image or URL"""
#     try:
#         # Check if image file was uploaded
#         if 'image' in request.files:
#             file = request.files['image']
#             img = Image.open(file.stream).convert('RGB')
#             embedding = generate_embedding(img)
#             return jsonify({
#                 'embedding': embedding,
#                 'dimension': len(embedding),
#                 'source': 'file'
#             })
        
#         # Check if image URL was provided
#         elif request.json and 'imageUrl' in request.json:
#             image_url = request.json['imageUrl']
#             response = requests.get(image_url, timeout=10)
#             img = Image.open(BytesIO(response.content)).convert('RGB')
#             embedding = generate_embedding(img)
#             return jsonify({
#                 'embedding': embedding,
#                 'dimension': len(embedding),
#                 'source': 'url'
#             })
        
#         # Check if base64 image was provided
#         elif request.json and 'imageBase64' in request.json:
#             image_data = base64.b64decode(request.json['imageBase64'].split(',')[1])
#             img = Image.open(BytesIO(image_data)).convert('RGB')
#             embedding = generate_embedding(img)
#             return jsonify({
#                 'embedding': embedding,
#                 'dimension': len(embedding),
#                 'source': 'base64'
#             })
        
#         else:
#             return jsonify({'error': 'No image provided'}), 400
            
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/batch-embeddings', methods=['POST'])
# def batch_embeddings():
#     """Generate embeddings for multiple image URLs"""
#     try:
#         data = request.json
#         image_urls = data.get('imageUrls', [])
        
#         if not image_urls:
#             return jsonify({'error': 'No image URLs provided'}), 400
        
#         results = []
#         for idx, url in enumerate(image_urls):
#             try:
#                 response = requests.get(url, timeout=10)
#                 img = Image.open(BytesIO(response.content)).convert('RGB')
#                 embedding = generate_embedding(img)
#                 results.append({
#                     'url': url,
#                     'embedding': embedding,
#                     'success': True
#                 })
#                 print(f"✅ Processed {idx + 1}/{len(image_urls)}")
#             except Exception as e:
#                 results.append({
#                     'url': url,
#                     'error': str(e),
#                     'success': False
#                 })
#                 print(f"❌ Failed {idx + 1}/{len(image_urls)}: {str(e)}")
        
#         return jsonify({
#             'results': results,
#             'total': len(image_urls),
#             'successful': sum(1 for r in results if r['success'])
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/cosine-similarity', methods=['POST'])
# def cosine_similarity():
#     """Calculate cosine similarity between two embeddings"""
#     try:
#         data = request.json
#         embedding1 = np.array(data['embedding1'])
#         embedding2 = np.array(data['embedding2'])
        
#         # Calculate cosine similarity
#         dot_product = np.dot(embedding1, embedding2)
#         norm1 = np.linalg.norm(embedding1)
#         norm2 = np.linalg.norm(embedding2)
        
#         similarity = dot_product / (norm1 * norm2)
        
#         return jsonify({
#             'similarity': float(similarity)
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
    
# #     print("📍 Server running at http://localhost:5000")
   
# #     app.run(host='0.0.0.0', port=5000, debug=True)


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
