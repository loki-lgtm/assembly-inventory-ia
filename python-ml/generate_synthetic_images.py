"""
Gerador de imagens sintéticas para demonstração do sistema
Cria imagens de peças industriais e caixas com etiquetas
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# Configurações
DATASET_DIR = "dataset"
BOXES_DIR = "boxes_dataset"
PIECES_TYPES = ["valvula", "retentor", "junta", "rolamento"]
COLORS = {
    "valvula": (59, 130, 246),      # Azul
    "retentor": (16, 185, 129),     # Verde
    "junta": (245, 158, 11),        # Laranja
    "rolamento": (139, 92, 246)     # Roxo
}

def create_directories():
    """Cria diretórios para o dataset"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(BOXES_DIR, exist_ok=True)
    for piece_type in PIECES_TYPES:
        os.makedirs(os.path.join(DATASET_DIR, piece_type), exist_ok=True)

def generate_piece_image(piece_type, index, size=(128, 128)):
    """Gera uma imagem sintética de uma peça"""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
    
    # Adiciona textura de fundo
    noise = np.random.randint(0, 30, (size[1], size[0], 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    center = (size[0] // 2, size[1] // 2)
    color = COLORS[piece_type]
    
    # Desenha a forma da peça
    if piece_type == "valvula":
        # Válvula: Retângulo com círculo no topo
        cv2.rectangle(img, (center[0] - 15, center[1] - 10), 
                     (center[0] + 15, center[1] + 40), color, -1)
        cv2.circle(img, (center[0], center[1] - 10), 20, color, -1)
        
    elif piece_type == "retentor":
        # Retentor: Círculo com buraco no meio
        cv2.circle(img, center, 35, color, -1)
        cv2.circle(img, center, 20, (240, 240, 240), -1)
        
    elif piece_type == "junta":
        # Junta: Anel fino
        cv2.circle(img, center, 40, color, 5)
        
    elif piece_type == "rolamento":
        # Rolamento: Círculo com linhas radiais
        cv2.circle(img, center, 35, color, -1)
        for angle in range(0, 360, 45):
            x = int(center[0] + 30 * np.cos(np.radians(angle)))
            y = int(center[1] + 30 * np.sin(np.radians(angle)))
            cv2.line(img, center, (x, y), (200, 200, 200), 2)
    
    # Adiciona sombra
    shadow = np.zeros_like(img)
    if piece_type == "valvula":
        cv2.rectangle(shadow, (center[0] - 13, center[1] - 8), 
                     (center[0] + 17, center[1] + 42), (0, 0, 0), -1)
        cv2.circle(shadow, (center[0] + 2, center[1] - 8), 20, (0, 0, 0), -1)
    else:
        cv2.circle(shadow, (center[0] + 2, center[1] + 2), 35, (0, 0, 0), -1)
    
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    img = cv2.addWeighted(img, 1.0, shadow, 0.3, 0)
    
    # Rotação aleatória
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, size)
    
    return img

def generate_box_image(box_code, index, size=(400, 300)):
    """Gera uma imagem sintética de uma caixa com etiqueta"""
    # Cria imagem base
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Desenha a caixa
    box_rect = [50, 50, size[0] - 50, size[1] - 50]
    draw.rectangle(box_rect, fill=(180, 140, 100), outline=(100, 70, 50), width=3)
    
    # Desenha a etiqueta
    label_rect = [size[0]//2 - 100, size[1]//2 - 40, size[0]//2 + 100, size[1]//2 + 40]
    draw.rectangle(label_rect, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Adiciona o código na etiqueta
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    # Centraliza o texto
    bbox = draw.textbbox((0, 0), box_code, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = size[0]//2 - text_width//2
    text_y = size[1]//2 - text_height//2
    
    draw.text((text_x, text_y), box_code, fill=(0, 0, 0), font=font)
    
    # Converte para numpy array
    img_np = np.array(img)
    
    # Adiciona ruído leve
    noise = np.random.randint(-10, 10, img_np.shape, dtype=np.int16)
    img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img_np

def generate_dataset():
    """Gera o dataset completo"""
    print("Gerando dataset de peças...")
    
    # Gera 15 imagens de cada tipo de peça
    for piece_type in PIECES_TYPES:
        print(f"  Gerando imagens de {piece_type}...")
        for i in range(15):
            img = generate_piece_image(piece_type, i)
            filename = os.path.join(DATASET_DIR, piece_type, f"{piece_type}_{i:03d}.jpg")
            cv2.imwrite(filename, img)
    
    print(f"\n✓ Dataset de peças criado em '{DATASET_DIR}/'")
    
    # Gera imagens de caixas
    print("\nGerando imagens de caixas...")
    box_codes = [
        "CX-VLV-001",
        "CX-RET-002",
        "CX-JNT-003",
        "CX-ROL-004",
        "CX-VLV-005"
    ]
    
    for i, code in enumerate(box_codes):
        img = generate_box_image(code, i)
        filename = os.path.join(BOXES_DIR, f"box_{i:03d}.jpg")
        cv2.imwrite(filename, img)
    
    print(f"✓ Imagens de caixas criadas em '{BOXES_DIR}/'")
    print(f"\nDataset completo gerado com sucesso!")
    print(f"  - Peças: {len(PIECES_TYPES) * 15} imagens")
    print(f"  - Caixas: {len(box_codes)} imagens")

if __name__ == "__main__":
    create_directories()
    generate_dataset()
