from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.storage import save_file
from app.schemas.document import DocumentCreate
from app.models.document import Document
from app.core.database import SessionLocal

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Formato não suportado")

    file_path = save_file(file)

    db = SessionLocal()
    document = Document(
        filename=file.filename,
        storage_path=file_path,
        status="uploaded"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return {
        "document_id": document.id,
        "status": document.status
    }
