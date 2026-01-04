import shutil
import os
import uuid
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_file(file: UploadFile) -> str:
    # Gerar um nome seguro se o filename vier vazio
    filename = file.filename if file.filename else f"doc_{uuid.uuid4()}.pdf"
    
    # Criar caminho completo
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Guardar o ficheiro
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_path