from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
import os

# Definição de caminhos
BASE_MODEL = "microsoft/trocr-base-handwritten"
LOCAL_MODEL = "models/ocr/trocr_finetuned_v1"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    Carrega o processador (sempre da base) e o modelo (local se existir).
    """
    print(f"🔧 A carregar processador base: {BASE_MODEL}")
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)

    if os.path.exists(LOCAL_MODEL):
        print(f"🧠 A carregar modelo treinado localmente (Português/Angola): {LOCAL_MODEL}")
        model = VisionEncoderDecoderModel.from_pretrained(LOCAL_MODEL)
    else:
        print(f"🌐 A carregar modelo base (Inglês): {BASE_MODEL}")
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    model.to(device)
    return processor, model

# Carregar recursos globais
processor, model = load_model()

def run_trocr(image_np: np.ndarray):
    """
    Recebe uma imagem (array numpy), converte e extrai o texto.
    """
    try:
        image = Image.fromarray(image_np).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # MUDANÇA AQUI: Adicionei num_beams=4 e penalidades
        # Isto ajuda o modelo a focar-se melhor e cometer menos erros de "alucinação"
        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,           # Tenta 4 caminhos diferentes (melhor qualidade)
            early_stopping=True,   # Para quando a frase estiver completa
            no_repeat_ngram_size=3 # Evita repetir palavras (ex: "Luanda Luanda")
        )
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception as e:
        print(f"❌ Erro na inferência do TrOCR: {e}")
        return ""