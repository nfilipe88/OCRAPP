from sqlalchemy import Column, Integer, ForeignKey, Text, DateTime
from sqlalchemy.sql import func
from app.models.base import Base

class CorrecaoHumana(Base):
    __tablename__ = "correcoes_humanas"

    id = Column(Integer, primary_key=True)
    segmento_id = Column(Integer, ForeignKey("ocr_segmentos.id"))
    texto_corrigido = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
