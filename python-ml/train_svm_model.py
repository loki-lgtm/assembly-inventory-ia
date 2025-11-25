"""
Script de treinamento do modelo SVM para identificação de peças industriais
Utiliza HOG (Histogram of Oriented Gradients) para extração de características
"""

import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Configurações
DATASET_PATH = "dataset"
MODEL_FILENAME = "svm_pieces_model.pkl"
IMAGE_SIZE = (64, 64)

def extract_hog_features(image):
    """Extrai características HOG de uma imagem"""
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    resized_image = cv2.resize(gray_image, IMAGE_SIZE)
    
    features = hog(
        resized_image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    return features

def load_dataset(dataset_path):
    """Carrega imagens do dataset e extrai features"""
    features_list = []
    labels_list = []
    
    print(f"Carregando dataset de: {dataset_path}")
    
    for label_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"  Processando classe: '{label_name}'")
        count = 0
        
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"    Aviso: Não foi possível ler {image_path}")
                continue
            
            hog_features = extract_hog_features(image)
            features_list.append(hog_features)
            labels_list.append(label_name)
            count += 1
        
        print(f"    {count} imagens carregadas")
    
    if not features_list:
        raise ValueError("Nenhuma imagem foi carregada. Verifique a estrutura do dataset.")
    
    return np.array(features_list), np.array(labels_list)

def train_and_evaluate():
    """Treina e avalia o modelo SVM"""
    try:
        # Carrega os dados
        print("\n" + "="*60)
        print("TREINAMENTO DO MODELO SVM - IDENTIFICAÇÃO DE PEÇAS")
        print("="*60 + "\n")
        
        features, labels = load_dataset(DATASET_PATH)
        print(f"\nTotal de {len(features)} imagens carregadas")
        print(f"Classes encontradas: {np.unique(labels)}")
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nDados divididos:")
        print(f"  Treino: {len(X_train)} amostras")
        print(f"  Teste: {len(X_test)} amostras")
        
        # Grid Search para otimização de hiperparâmetros
        print("\nIniciando treinamento com otimização de hiperparâmetros...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        print("\n" + "="*60)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*60)
        print(f"\nMelhores parâmetros: {grid_search.best_params_}")
        print(f"Melhor score (CV): {grid_search.best_score_:.4f}")
        
        # Avalia no conjunto de teste
        print("\nAvaliando modelo no conjunto de teste...")
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nAcurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "="*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*60)
        print(classification_report(y_test, predictions))
        
        # Salva o modelo
        print(f"\nSalvando modelo em '{MODEL_FILENAME}'...")
        joblib.dump(best_model, MODEL_FILENAME)
        
        print("\n" + "="*60)
        print("✓ MODELO SALVO COM SUCESSO!")
        print("="*60)
        print(f"\nO sistema está pronto para identificar peças industriais.")
        print(f"Arquivo do modelo: {MODEL_FILENAME}")
        
        return best_model
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = train_and_evaluate()
    if model:
        print("\n✓ Treinamento concluído com sucesso!")
    else:
        print("\n❌ Falha no treinamento")
