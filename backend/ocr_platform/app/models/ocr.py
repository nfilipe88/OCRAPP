from sqlalchemy import ForeignKey, Text, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class OCRResultado(Base):
    __tablename__ = "ocr_resultados"

    # FIX: Usar Mapped[...] diz ao Pylance o tipo real da variável
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"))
    texto_completo: Mapped[str] = mapped_column(Text, nullable=True)
    confidence_global: Mapped[float] = mapped_column(Float, default=0.0)