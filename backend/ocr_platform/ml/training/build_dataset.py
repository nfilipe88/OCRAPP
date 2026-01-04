from app.core.database import SessionLocal
from app.models.segment import OCRSegmento
from app.models.correction import CorrecaoHumana

def build_training_dataset():
    db = SessionLocal()

    data = []

    corrections = db.query(CorrecaoHumana).all()

    for correction in corrections:
        segment = db.query(OCRSegmento).get(correction.segmento_id)

        data.append({
            "image_path": segment.imagem_path,
            "text": correction.texto_corrigido
        })

    print(f"Dataset criado com {len(data)} exemplos")
    return data
