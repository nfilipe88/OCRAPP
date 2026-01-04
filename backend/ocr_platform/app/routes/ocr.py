from fastapi import APIRouter
from app.core.database import SessionLocal
from app.models.segment import OCRSegmento
from app.models.correction import CorrecaoHumana

router = APIRouter()

@router.get("/segments/{document_id}")
def get_segments(document_id: int):
    db = SessionLocal()
    return db.query(OCRSegmento).all()

@router.post("/segments/{segment_id}/correct")
def correct_segment(segment_id: int, texto_corrigido: str):
    db = SessionLocal()

    correction = CorrecaoHumana(
        segmento_id=segment_id,
        texto_corrigido=texto_corrigido
    )
    db.add(correction)
    db.commit()

    return {"status": "correction_saved"}
