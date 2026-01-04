from app.core.database import engine
from app.models.base import Base
# Importar todos os modelos para o SQLAlchemy os reconhecer
from app.models.document import Document
from app.models.ocr import OCRResultado
from app.models.segment import OCRSegmento
from app.models.correction import CorrecaoHumana

print("A criar tabelas na base de dados...")
Base.metadata.create_all(bind=engine)
print("Tabelas criadas com sucesso!")