import sys
import os

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
    print(f"Iniciando processamento do documento {document_id}...")
    db = SessionLocal()
    
    try:
        # CORREÇÃO: Usar db.get() em vez de db.query().get() (Moderno)
        document = db.get(Document, document_id)
        
        if not document:
            print("Documento não encontrado.")
            return

        print(f"A ler ficheiro: {document.storage_path}")

        # 1. Pré-processamento (Agora suporta PDF!)
        image = preprocess_image(document.storage_path)
        lines = segment_lines(image)
        print(f"Documento segmentado em {len(lines)} linhas.")

        # 2. Criar registo do resultado
        ocr_result = OCRResultado(
            document_id=document.id,
            texto_completo="",
            confidence_global=0.0
        )
        db.add(ocr_result)
        db.commit()
        db.refresh(ocr_result)

        full_text = []
        confidences = []
        
        os.makedirs("segments", exist_ok=True)

        # 3. Processar cada linha
        print("A iniciar inferência com TrOCR...")
        for i, line in enumerate(lines):
            try:
                text = run_trocr(line)
                full_text.append(text)
                confidences.append(1) # Simulação

                seg_filename = f"segments/{document.id}_{i}_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(seg_filename, line)

                segment = OCRSegmento(
                    ocr_resultado_id=ocr_result.id,
                    imagem_path=seg_filename,
                    texto_previsto=text,
                    confidence=1
                )
                db.add(segment)
                # print(f"Linha {i}: {text}") # Descomenta para ver em tempo real
                
            except Exception as e:
                print(f"Erro na linha {i}: {e}")

        # 4. Finalizar
        ocr_result.texto_completo = "\n".join(full_text)
        ocr_result.confidence_global = sum(confidences) / len(confidences) if confidences else 0.0
        
        document.status = "ocr_completed"
        db.commit()
        print("Processamento concluído com sucesso! 🚀")

    except Exception as e:
        print(f"Erro crítico ao processar: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    process_document(1)