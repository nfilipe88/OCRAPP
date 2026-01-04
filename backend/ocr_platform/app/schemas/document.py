from pydantic import BaseModel
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    status: str
    storage_path: str
    created_at: datetime

    class Config:
        from_attributes = True  # Permite ler dados do modelo SQLAlchemy