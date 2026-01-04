import cv2
import numpy as np
import fitz  # PyMuPDF

def load_image_or_pdf(path: str) -> np.ndarray:
    """Carrega PDF ou Imagem com alta resolução."""
    if path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(path)
            # Zoom 2.0x é crucial para ver traços finos
            mat = fitz.Matrix(2.0, 2.0)
            page = doc[0]
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 3: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            return img_array
        except Exception as e:
            raise ValueError(f"Erro ao converter PDF: {e}")
    else:
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Não foi possível ler: {path}")
        return img

def normalize_background(image):
    """
    Remove o fundo do papel (amarelo/ruído), deixando-o branco.
    Ideal para documentos históricos.
    """
    # 1. Converter para escala de cinzentos
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Morfologia para estimar o fundo (ignora letras, vê só o papel)
    # Kernel grande (25x25) para não apagar letras grandes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 3. Divisão (Normalização)
    # Ao dividir a imagem pelo fundo, tudo o que é fundo vira branco (255)
    # E o que é tinta escura mantém-se escuro.
    normalized = cv2.divide(gray, background, scale=255)

    # 4. Binarização Limpa (Otsu)
    # Agora que o fundo é uniforme, o Otsu não falha.
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary, normalized

def remove_lines_and_borders(binary_image):
    """Remove linhas de tabelas e bordas pretas de digitalização."""
    h, w = binary_image.shape
    
    # 1. Apagar bordas pretas (Scan noise)
    # Cria uma moldura branca de 30px à volta
    cv2.rectangle(binary_image, (0,0), (w, 30), 0, -1)      # Topo
    cv2.rectangle(binary_image, (0, h-30), (w, h), 0, -1)   # Fundo
    cv2.rectangle(binary_image, (0,0), (30, h), 0, -1)      # Esquerda
    cv2.rectangle(binary_image, (w-30, 0), (w, h), 0, -1)   # Direita

    # 2. Remover linhas horizontais longas (pautas)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_h = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel)
    
    # 3. Remover linhas verticais (tabelas)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_v = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, v_kernel)
    
    # Subtrair tudo
    clean = cv2.subtract(binary_image, detected_h)
    clean = cv2.subtract(clean, detected_v)
    
    # Limpeza final de "poeira" (pontos isolados)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
    return clean

def ensure_horizontal_orientation(binary_image, original_image):
    """Garante que o texto está na horizontal."""
    proj_0 = np.sum(binary_image, axis=1)
    var_0 = np.var(proj_0)

    rotated_bin_90 = cv2.rotate(binary_image, cv2.ROTATE_90_CLOCKWISE)
    proj_90 = np.sum(rotated_bin_90, axis=1)
    var_90 = np.var(proj_90)

    print(f"   -> Variância 0º: {var_0:.0f} | Variância 90º: {var_90:.0f}")

    # Se a variância a 90º for muito maior, roda.
    if var_90 > var_0 * 1.3:
        print("   -> Rotação: Documento deitado detetado. A rodar 90º.")
        rotated_img = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_bin_90, rotated_img
    
    return binary_image, original_image

def correct_small_skew(image, binary_image):
    """Corrige inclinações subtis (<10º)."""
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) == 0: return image, binary_image
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle

    if abs(angle) > 0.5 and abs(angle) < 10:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bin_img = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, bin_img
    
    return image, binary_image

def preprocess_image(image_path: str) -> np.ndarray:
    # 1. Carregar
    img = load_image_or_pdf(image_path)
    
    # 2. Normalizar Fundo (Técnica nova)
    binary, _ = normalize_background(img)
    
    # 3. Limpezas (Bordas pretas e linhas de tabela)
    clean_binary = remove_lines_and_borders(binary)
    
    # 4. Rotação e Orientação
    clean_binary, img = ensure_horizontal_orientation(clean_binary, img)
    img, clean_binary = correct_small_skew(img, clean_binary)

    return clean_binary

def segment_lines(image: np.ndarray):
    """Segmentação robusta com threshold dinâmico."""
    # Dilatação para unir palavras
    width_kernel = max(25, int(image.shape[1] * 0.03)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_kernel, 1))
    dilated = cv2.dilate(image, kernel, iterations=1)

    projection = np.sum(dilated, axis=1)
    
    # Threshold dinâmico: 20% da média de tinta encontrada
    # Isto adapta-se se o documento tiver muita ou pouca escrita
    non_zero_proj = projection[projection > 0]
    if len(non_zero_proj) > 0:
        threshold = np.mean(non_zero_proj) * 0.2
    else:
        threshold = 1
    
    lines = []
    start = None
    margin = 5

    for i, value in enumerate(projection):
        if value > threshold and start is None:
            start = max(0, i - margin)
        elif value <= threshold and start is not None:
            end = min(image.shape[0], i + margin)
            
            roi = image[start:end, :]
            
            # Filtro de qualidade rigoroso
            # Ignora manchas pequenas (<20px altura) ou sem tinta suficiente
            if roi.shape[0] > 20 and roi.shape[1] > 50 and cv2.countNonZero(roi) > 100:
                lines.append(cv2.bitwise_not(roi))
            
            start = None

    if start is not None:
        lines.append(cv2.bitwise_not(image[start:, :]))

    return lines