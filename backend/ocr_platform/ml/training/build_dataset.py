import sys
import os
sys.path.append(os.getcwd())
import csv
import shutil
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.correction import CorrecaoHumana
from app.models.segment import OCRSegmento
from sklearn.model_selection import train_test_split


def build_dataset(output_dir="data/dataset_v1", test_size=0.1):
    """
    Exporta as correções humanas da BD para uma pasta de dataset organizada.
    Estrutura:
    /data/dataset_v1/
       /images/
       metadata.csv (file_name, text)
    """
    print(f"🔨 A construir dataset em '{output_dir}'...")
    
    db = SessionLocal()
    
    # 1. Buscar todas as correções humanas
    # Fazemos join com o Segmento para saber qual é a imagem
    corrections = db.query(CorrecaoHumana, OCRSegmento)\
        .join(OCRSegmento, CorrecaoHumana.segmento_id == OCRSegmento.id)\
        .all()

    if not corrections:
        print("⚠️ Nenhuma correção encontrada na base de dados. Valida alguns segmentos primeiro!")
        return

    print(f"📚 Encontrados {len(corrections)} exemplos validados.")

    # 2. Preparar pastas
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    data_entries = []

    # 3. Processar cada correção
    for correction, segment in corrections:
        # Nome do ficheiro original
        original_path = segment.imagem_path
        
        if not os.path.exists(original_path):
            print(f"❌ Imagem não encontrada: {original_path} (saltando...)")
            continue

        # Copiar imagem para a pasta do dataset (para ficar tudo junto)
        filename = os.path.basename(original_path)
        target_path = os.path.join(images_dir, filename)
        shutil.copy2(original_path, target_path)

        # Adicionar à lista
        data_entries.append({
            "file_name": filename,
            "text": correction.texto_corrigido
        })

    # 4. Dividir em Treino e Teste (90% aprender, 10% validar)
    # Se houver poucos dados, usamos tudo para treino
    if len(data_entries) > 5:
        train_data, test_data = train_test_split(data_entries, test_size=test_size, random_state=42)
    else:
        train_data = data_entries
        test_data = []

    # 5. Guardar metadados (CSV)
    save_csv(train_data, os.path.join(output_dir, "train.csv"))
    save_csv(test_data, os.path.join(output_dir, "test.csv"))
    
    print(f"✅ Dataset construído com sucesso!")
    print(f"   - Treino: {len(train_data)} exemplos")
    print(f"   - Teste: {len(test_data)} exemplos")

def save_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        for item in data:
            writer.writerow([item["file_name"], item["text"]])

if __name__ == "__main__":
    # Precisamos instalar scikit-learn: pip install scikit-learn
    build_dataset()