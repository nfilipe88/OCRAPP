from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo e processador
# MUDAR DE:
# processor = TrOCRProcessor.from_pretrained("models/ocr/trocr_finetuned_v1")
# model = VisionEncoderDecoderModel.from_pretrained("models/ocr/trocr_finetuned_v1")

# PARA:
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(device) # Mover para device numa linha separada para ajudar o Pylance

def run_trocr(image_np: np.ndarray):
    # Converter array numpy para Imagem PIL
    image = Image.fromarray(image_np).convert("RGB")
    
    # FIX: Usar 'images=' explicitamente para satisfazer o Pylance
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # FIX: Chamar generate garantindo que o modelo é tratado como tal
    pixel_values = pixel_values.to(device)        # move inputs
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)    # do not pass device here
    
    # Descodificar
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text