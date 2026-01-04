from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, ocr
import os

app = FastAPI(
    title="OCR Inteligente – Registos de Angola",
    version="0.1.0"
)

# 1. Configurar CORS
# Isto permite que o ficheiro HTML (que vamos criar) consiga falar com a API
# sem que o navegador bloqueie por segurança.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringe-se isto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Criar pastas necessárias (segurança extra)
os.makedirs("segments", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# 3. Montar a pasta de segmentos como "estática"
# Agora, se acederes a /segments/imagem.png, a API devolve a imagem
app.mount("/segments", StaticFiles(directory="segments"), name="segments")

# 4. Registar Rotas
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])