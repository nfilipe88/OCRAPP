import sys
import os

# Adiciona a raiz do projeto ao path (para evitar erros de módulo)
sys.path.append(os.getcwd())

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # <--- MUDANÇA AQUI: Importar do PyTorch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

class OCRDataset(Dataset):
    """Classe que carrega as imagens e textos para o PyTorch"""
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Ler nome do ficheiro e texto do CSV
        file_name = self.df.iloc[idx]['file_name']
        text = str(self.df.iloc[idx]['text']) # Garantir que é string
        
        # Carregar imagem
        image_path = os.path.join(self.root_dir, file_name)
        # Importante: converter para RGB
        image = Image.open(image_path).convert("RGB")
        
        # Preparar imagem (converter para números/tensores)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Preparar texto (converter para IDs de tokens)
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True # Importante para evitar erros de tamanho
        ).input_ids
        
        # Substituir o token de padding por -100
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def train():
    # --- Configurações ---
    DATASET_DIR = "data/dataset_v1"
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
    
    # Modelo base
    MODEL_NAME = "microsoft/trocr-base-handwritten"
    
    # Onde guardar o modelo afinado
    OUTPUT_DIR = "models/ocr/trocr_finetuned_v1"
    
    # Parâmetros de Treino
    BATCH_SIZE = 2    
    EPOCHS = 10       
    LEARNING_RATE = 5e-5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 A iniciar treino em: {device}")

    # 1. Carregar Dados
    if not os.path.exists(TRAIN_CSV):
        print("❌ Ficheiro train.csv não encontrado. Corre o build_dataset.py primeiro.")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.dropna()
    print(f"📚 Exemplos de treino válidos: {len(train_df)}")

    if len(train_df) == 0:
        print("⚠️ O dataset está vazio!")
        return

    # 2. Preparar Modelo e Processador
    print("🧠 A carregar modelo TrOCR...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Configurações técnicas
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.to(device)

    # 3. Criar DataLoader
    train_dataset = OCRDataset(root_dir=IMAGES_DIR, df=train_df, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Otimizador (Agora usa o do PyTorch)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Loop de Treino
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"📉 Epoch {epoch+1} | Média Loss: {avg_loss:.4f}")

    # 6. Guardar o Modelo
    print(f"💾 A guardar o novo modelo em '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Treino concluído com sucesso!")

if __name__ == "__main__":
    train()