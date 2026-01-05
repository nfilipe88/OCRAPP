import sys
import os
import time

# Adiciona o diretório raiz ao path
sys.path.append(os.getcwd())

from ml.preprocessing.image import preprocess_image, segment_lines
from ml.inference.trocr import run_trocr
from app.models.ocr import OCRResultado
from app.models.segment import OCRSegmento
from app.models.document import Document
from app.core.database import SessionLocal
import uuid
import cv2

def process_document(document_id: int):
    print(f"▶️ A iniciar processamento do documento {document_id}...")
    db = SessionLocal()
    
    try:
        document = db.get(Document, document_id)
        
        if not document:
            print(f"❌ Documento {document_id} não encontrado.")
            return

        # Bloquear o documento (status 'processing') para ninguém mais mexer
        document.status = "processing"
        db.commit()

        print(f"   📂 Ficheiro: {document.storage_path}")

        # 1. Pré-processamento (Limpeza e Segmentação)
        # O teu image.py já tem a limpeza de fundo e rotação automática
        image = preprocess_image(document.storage_path)
        lines = segment_lines(image)
        print(f"   ✂️ Documento segmentado em {len(lines)} linhas.")

        # 2. Criar registo de OCR
        ocr_result = OCRResultado(
            document_id=document.id,
            texto_completo="",
            confidence_global=0.0
        )
        db.add(ocr_result)
        db.commit()
        db.refresh(ocr_result)

        full_text = []
        
        os.makedirs("segments", exist_ok=True)

        # 3. Leitura com IA
        print("   🧠 A ler linhas com TrOCR...")
        for i, line in enumerate(lines):
            try:
                # O run_trocr agora já carrega o teu modelo fino automaticamente!
                text = run_trocr(line)
                
                if not text.strip():
                    continue

                full_text.append(text)

                # Guardar o recorte para validação no frontend
                seg_filename = f"segments/{document.id}_{i}_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(seg_filename, line)

                segment = OCRSegmento(
                    ocr_resultado_id=ocr_result.id,
                    imagem_path=seg_filename,
                    texto_previsto=text,
                    confidence=1.0 
                )
                db.add(segment)
                
            except Exception as e:
                print(f"   ⚠️ Erro na linha {i}: {e}")

        # 4. Finalizar
        ocr_result.texto_completo = "\n".join(full_text)
        ocr_result.confidence_global = 1.0
        
        document.status = "ocr_completed"
        db.commit()
        print(f"✅ Documento {document_id} concluído com sucesso!")

    except Exception as e:
        print(f"❌ Erro crítico ao processar {document_id}: {e}")
        # Marcar como erro para não ficar preso em 'processing' para sempre
        try:
            document.status = "error"
            db.commit()
        except:
            pass
        db.rollback()
    finally:
        db.close()

def start_worker():
    print("👷 OCR Worker Automático iniciado! A aguardar documentos...")
    print("   (Pressione Ctrl+C para parar)")
    
    while True:
        db = SessionLocal()
        try:
            # Procura o documento mais antigo que ainda esteja como 'uploaded'
            pending_doc = db.query(Document)\
                .filter(Document.status == "uploaded")\
                .order_by(Document.created_at.asc())\
                .first()

            if pending_doc:
                # Encontrou trabalho! Mãos à obra.
                process_document(pending_doc.id)
            else:
                # Não há trabalho? Dorme 2 segundos e tenta de novo.
                time.sleep(2)
                
        except Exception as e:
            print(f"⚠️ Erro no ciclo do worker: {e}")
            time.sleep(5)
        finally:
            db.close()

if __name__ == "__main__":
    start_worker()