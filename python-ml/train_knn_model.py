"""
Script de treinamento do modelo KNN para localização de caixas
Combina OCR para leitura de códigos com KNN para reconhecimento de padrões
"""

import os
import cv2
import joblib
import numpy as np
import pytesseract
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configurações
BOXES_DIR = "boxes_dataset"
MODEL_FILENAME = "knn_boxes_model.pkl"
IMAGE_SIZE = (200, 150)

# Mapeamento de códigos para localizações
BOX_LOCATIONS = {
    "CX-VLV-001": "A-1-03",
    "CX-RET-002": "A-2-05",
    "CX-JNT-003": "B-1-02",
    "CX-ROL-004": "B-3-04",
    "CX-VLV-005": "C-2-01"
}

def preprocess_for_ocr(image):
    """Pré-processa imagem para OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_box_features(image):
    """Extrai características da imagem da caixa"""
    # Redimensiona
    resized = cv2.resize(image, IMAGE_SIZE)
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Calcula histograma
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-7)  # Normaliza
    
    # Calcula gradientes
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(gx, gy)
    
    # Estatísticas dos gradientes
    grad_features = [
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.min(mag)
    ]
    
    # Combina features
    features = np.concatenate([hist, grad_features])
    
    return features

def read_code_from_image(image):
    """Lê código da imagem usando OCR"""
    processed = preprocess_for_ocr(image)
    
    # Configuração do Tesseract
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --psm 6'
    
    text = pytesseract.image_to_string(processed, config=custom_config)
    
    # Limpa o texto
    text = text.strip().replace(" ", "").replace("\n", "")
    
    # Procura por padrão de código
    import re
    match = re.search(r'[A-Z]{2,3}-[A-Z]{3}-\d{3}', text)
    
    if match:
        return match.group(0)
    
    return text if text else "UNKNOWN"

def load_boxes_dataset():
    """Carrega dataset de caixas"""
    features_list = []
    codes_list = []
    
    print(f"Carregando imagens de caixas de: {BOXES_DIR}")
    
    if not os.path.exists(BOXES_DIR):
        raise ValueError(f"Diretório {BOXES_DIR} não encontrado")
    
    for image_name in sorted(os.listdir(BOXES_DIR)):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        image_path = os.path.join(BOXES_DIR, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"  Aviso: Não foi possível ler {image_path}")
            continue
        
        # Extrai features
        features = extract_box_features(image)
        
        # Lê código
        code = read_code_from_image(image)
        
        features_list.append(features)
        codes_list.append(code)
        
        print(f"  {image_name}: código '{code}' detectado")
    
    return np.array(features_list), np.array(codes_list)

def train_knn_model():
    """Treina modelo KNN"""
    try:
        print("\n" + "="*60)
        print("TREINAMENTO DO MODELO KNN - LOCALIZAÇÃO DE CAIXAS")
        print("="*60 + "\n")
        
        # Carrega dados
        features, codes = load_boxes_dataset()
        
        print(f"\nTotal de {len(features)} imagens carregadas")
        print(f"Códigos detectados: {np.unique(codes)}")
        
        # Como temos poucas amostras, vamos usar validação cruzada simples
        # Em produção, você teria mais dados
        
        print("\nTreinando modelo KNN...")
        
        # Treina KNN com k=3
        knn = KNeighborsClassifier(n_neighbors=min(3, len(features)), weights='distance')
        knn.fit(features, codes)
        
        # Testa no próprio conjunto (apenas para demonstração)
        predictions = knn.predict(features)
        accuracy = accuracy_score(codes, predictions)
        
        print(f"\nAcurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Salva o modelo
        print(f"\nSalvando modelo em '{MODEL_FILENAME}'...")
        model_data = {
            'knn': knn,
            'locations': BOX_LOCATIONS
        }
        joblib.dump(model_data, MODEL_FILENAME)
        
        print("\n" + "="*60)
        print("✓ MODELO KNN SALVO COM SUCESSO!")
        print("="*60)
        print(f"\nO sistema está pronto para localizar caixas.")
        print(f"Arquivo do modelo: {MODEL_FILENAME}")
        
        # Mostra mapeamento de localizações
        print("\nMapeamento de localizações:")
        for code, location in BOX_LOCATIONS.items():
            print(f"  {code} → {location}")
        
        return knn
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = train_knn_model()
    if model:
        print("\n✓ Treinamento concluído com sucesso!")
    else:
        print("\n❌ Falha no treinamento")
