import cv2
import numpy as np
import fitz  # PyMuPDF

def load_image_or_pdf(path: str) -> np.ndarray:
    if path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(path)
            # Zoom 2.0x para boa resolução
            mat = fitz.Matrix(2.0, 2.0)
            page = doc[0]
            pix = page.get_pixmap(matrix=mat)
            
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
            return img_array
        except Exception as e:
            raise ValueError(f"Erro ao converter PDF: {e}")
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Não foi possível ler a imagem: {path}")
        return img

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Prepara a imagem: Binarização Adaptativa + Remoção de Linhas Verticais/Horizontais.
    """
    img = load_image_or_pdf(image_path)
    
    # 1. Escala de Cinzentos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Binarização Adaptativa (Texto Branco, Fundo Preto)
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 
        10
    )
    
    # 3. Remover Linhas Verticais (Crucial para Formulários)
    # Cria um filtro que só deteta coisas altas e finas (linhas verticais)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Subtrai as linhas verticais da imagem original
    clean_binary = cv2.subtract(binary, detected_vertical)
    
    # 4. Remoção de Ruído restante
    kernel = np.ones((2,2), np.uint8)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel)

    return clean_binary

def segment_lines(image: np.ndarray):
    """Segmenta a imagem binária limpa em linhas de texto."""
    
    # 1. Dilatação Horizontal para fundir letras em linhas
    # Usamos um kernel largo (40,1) para conectar palavras
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(image, kernel, iterations=1)

    # 2. Projeção Horizontal
    projection = np.sum(dilated, axis=1)

    lines = []
    start = None
    margin = 5
    
    # Ajuste de sensibilidade: 2% do máximo (era 5%) para apanhar linhas mais finas
    threshold = np.max(projection) * 0.02

    for i, value in enumerate(projection):
        if value > threshold and start is None:
            start = max(0, i - margin)
        elif value <= threshold and start is not None:
            end = min(image.shape[0], i + margin)
            
            roi = image[start:end, :]
            
            # Filtro: Ignorar linhas muito pequenas (ruído) ou muito finas
            if roi.shape[0] > 15 and roi.shape[1] > 50:
                # Inverte para o formato que a IA gosta (Preto no Branco)
                roi_inverted = cv2.bitwise_not(roi)
                lines.append(roi_inverted)
            
            start = None

    if start is not None:
        lines.append(cv2.bitwise_not(image[start:, :]))

    return lines