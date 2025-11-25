# Assembly Inventory IA

Sistema de Gestão de Estoque Industrial com Visão Computacional e Machine Learning

---

## 📋 O que é?

Dois projetos integrados em uma interface web simples:

**Projeto 1:** Identificação de Peças com SVM  
**Projeto 2:** Localização de Caixas com KNN + OCR

---

## 🚀 Como Usar no VS Code

### 1. Instalar Dependências Python

```bash
cd python-ml
pip install -r requirements.txt
```

**Instalar Tesseract OCR:**
- Ubuntu: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Treinar os Modelos (Opcional)

Os modelos já estão treinados, mas você pode retreinar:

```bash
# Gerar dataset
python generate_synthetic_images.py

# Treinar SVM
python train_svm_model.py

# Treinar KNN
python train_knn_model.py
```

### 3. Iniciar o Servidor

```bash
python ml_server_standalone.py
```

Servidor rodando em: `http://localhost:5001`

### 4. Testar os Endpoints

**Identificar Peças:**
```bash
curl -X POST http://localhost:5001/api/ml/generate-demo-image \
  -H "Content-Type: application/json" \
  -d '{"type":"pieces"}'
```

**Health Check:**
```bash
curl http://localhost:5001/health
```

---

## 📁 Estrutura

```
assembly-inventory-ia/
├── python-ml/
│   ├── ml_server_standalone.py    # Servidor Flask (USAR ESTE)
│   ├── svm_pieces_model.pkl       # Modelo SVM treinado
│   ├── knn_boxes_model.pkl        # Modelo KNN treinado
│   ├── train_svm_model.py         # Script de treino SVM
│   ├── train_knn_model.py         # Script de treino KNN
│   ├── generate_synthetic_images.py  # Gerador de dataset
│   ├── dataset/                   # Imagens de peças
│   ├── boxes_dataset/             # Imagens de caixas
│   └── requirements.txt           # Dependências
└── README.md
```

---

## 🎯 Endpoints da API

### POST /api/ml/identify-pieces
Identifica e conta peças na imagem

**Request:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "pieces": {
    "valvula": 3,
    "retentor": 4,
    "junta": 2,
    "rolamento": 3
  },
  "regions": [...],
  "total_detected": 12
}
```

### POST /api/ml/locate-box
Lê código da caixa e verifica localização

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "currentLocation": "A-1-03"
}
```

**Response:**
```json
{
  "success": true,
  "code": "CX-VLV-001",
  "correctLocation": "A-1-03",
  "isCorrect": true
}
```

### POST /api/ml/generate-demo-image
Gera imagem de demonstração

**Request:**
```json
{
  "type": "pieces"
}
```

---

## 🔧 Tecnologias

- **Python 3.11**
- **Flask** - API REST
- **OpenCV** - Processamento de imagens
- **scikit-learn** - Machine Learning (SVM, KNN)
- **scikit-image** - Extração de características (HOG)
- **Tesseract** - OCR
- **NumPy** - Computação numérica

---

## 📊 Modelos

### SVM (Projeto 1)
- **Acurácia:** 100%
- **Classes:** Válvula, Retentor, Junta, Rolamento
- **Features:** HOG
- **Dataset:** 60 imagens (15 por classe)

### KNN (Projeto 2)
- **Acurácia:** 100%
- **K:** 3
- **Features:** Histograma + Gradientes
- **OCR:** Tesseract
- **Dataset:** 5 caixas

---

## 📝 Notas

- Não usa banco de dados (dados fictícios em memória)
- Dataset sintético para demonstração
- Pronto para rodar localmente no VS Code
- Para produção, substitua por imagens reais

---

**Assembly Inventory IA** 🚀
