from sqlalchemy import ForeignKey, String, Text, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class OCRSegmento(Base):
    __tablename__ = "ocr_segmentos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ocr_resultado_id: Mapped[int] = mapped_column(Integer, ForeignKey("ocr_resultados.id"))
    imagem_path: Mapped[str] = mapped_column(String)
    texto_previsto: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)