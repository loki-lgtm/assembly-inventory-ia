"""
Assembly Inventory IA - Servidor ML Standalone
Versão simplificada sem banco de dados, usando dados fictícios
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import base64
import re
from skimage.feature import hog
import pytesseract

app = Flask(__name__)
CORS(app)

# Carrega os modelos treinados
print("Carregando modelos...")
svm_model = joblib.load('svm_pieces_model.pkl')
knn_data = joblib.load('knn_boxes_model.pkl')
knn_model = knn_data['knn']
box_locations = knn_data['locations']
print("✓ Modelos carregados!")

# Configurações
IMAGE_SIZE_PIECES = (64, 64)
IMAGE_SIZE_BOXES = (200, 150)
CONFIDENCE_THRESHOLD = 0.6

# Cores para cada tipo de peça
PIECE_COLORS = {
    "valvula": "#3B82F6",
    "retentor": "#10B981",
    "junta": "#F59E0B",
    "rolamento": "#8B5CF6"
}

# Dados fictícios de estoque
INVENTORY_DATA = {
    "valvula": {"name": "Válvula", "stock": 45, "min": 20},
    "retentor": {"name": "Retentor", "stock": 32, "min": 25},
    "junta": {"name": "Junta", "stock": 18, "min": 30},
    "rolamento": {"name": "Rolamento", "stock": 28, "min": 15}
}

def decode_base64_image(base64_string):
    """Decodifica imagem base64"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_hog_features(image):
    """Extrai características HOG"""
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    resized_image = cv2.resize(gray_image, IMAGE_SIZE_PIECES)
    
    features = hog(
        resized_image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    return features

def segment_pieces(image):
    """Segmenta peças individuais"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    return regions

def read_code_from_image(image):
    """Lê código via OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    text = text.strip().replace(" ", "").replace("\n", "")
    
    match = re.search(r'[A-Z]{2,3}-[A-Z]{3}-\d{3}', text)
    if match:
        return match.group(0)
    
    return text if text else "UNKNOWN"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({'status': 'ok', 'message': 'Assembly Inventory IA - ML Server Running'})

@app.route('/api/ml/identify-pieces', methods=['POST'])
def identify_pieces():
    """Identifica peças na imagem"""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_base64_image(image_base64)
        regions = segment_pieces(image)
        
        detected_pieces = {}
        classified_regions = []
        
        for region in regions:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            roi = image[y:y+h, x:x+w]
            
            features = extract_hog_features(roi)
            probabilities = svm_model.predict_proba([features])[0]
            best_idx = probabilities.argmax()
            confidence = probabilities[best_idx]
            
            if confidence > CONFIDENCE_THRESHOLD:
                piece_type = svm_model.classes_[best_idx]
                detected_pieces[piece_type] = detected_pieces.get(piece_type, 0) + 1
                
                classified_regions.append({
                    **region,
                    'type': piece_type,
                    'confidence': float(confidence),
                    'color': PIECE_COLORS.get(piece_type, '#888888')
                })
        
        return jsonify({
            'success': True,
            'pieces': detected_pieces,
            'regions': classified_regions,
            'total_detected': len(classified_regions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/locate-box', methods=['POST'])
def locate_box():
    """Localiza caixa e lê código"""
    try:
        data = request.json
        image_base64 = data.get('image')
        current_location = data.get('currentLocation', '')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_base64_image(image_base64)
        code = read_code_from_image(image)
        
        correct_location = box_locations.get(code, 'DESCONHECIDA')
        is_correct = (current_location.upper() == correct_location.upper()) if current_location else None
        
        return jsonify({
            'success': True,
            'code': code,
            'correctLocation': correct_location,
            'currentLocation': current_location,
            'isCorrect': is_correct,
            'content': f"Conteúdo da caixa {code}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/generate-demo-image', methods=['POST'])
def generate_demo_image():
    """Gera imagem de demonstração"""
    try:
        data = request.json
        demo_type = data.get('type', 'pieces')
        
        if demo_type == 'pieces':
            img = np.ones((480, 640, 3), dtype=np.uint8) * 240
            noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            pieces_count = {'valvula': 3, 'retentor': 4, 'junta': 2, 'rolamento': 3}
            
            for piece_type, count in pieces_count.items():
                color_hex = PIECE_COLORS[piece_type]
                color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                
                for _ in range(count):
                    x = np.random.randint(50, 590)
                    y = np.random.randint(50, 430)
                    size = np.random.randint(20, 35)
                    
                    if piece_type == 'valvula':
                        cv2.rectangle(img, (x-10, y-5), (x+10, y+30), color_bgr, -1)
                        cv2.circle(img, (x, y-5), 15, color_bgr, -1)
                    elif piece_type == 'retentor':
                        cv2.circle(img, (x, y), size, color_bgr, -1)
                        cv2.circle(img, (x, y), size//2, (240, 240, 240), -1)
                    elif piece_type == 'junta':
                        cv2.circle(img, (x, y), size, color_bgr, 4)
                    else:
                        cv2.circle(img, (x, y), size, color_bgr, -1)
            
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'expectedPieces': pieces_count
            })
        
        else:
            return jsonify({'error': 'Invalid demo type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    """Retorna dados fictícios de estoque"""
    return jsonify(INVENTORY_DATA)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Assembly Inventory IA - ML Server")
    print("="*60)
    print("Endpoints:")
    print("  POST /api/ml/identify-pieces")
    print("  POST /api/ml/locate-box")
    print("  POST /api/ml/generate-demo-image")
    print("  GET  /api/inventory")
    print("  GET  /health")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
