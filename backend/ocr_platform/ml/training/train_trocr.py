from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class HandwritingDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]

        # FIX: Argumento nomeado 'images'
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text, return_tensors="pt"
        ).input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}


def train_model(dataset_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # FIX: Separar a inicialização do movimento para a GPU
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to(device)

    dataset = HandwritingDataset(dataset_data, processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        loss_val = 0.0
        for batch in loader:
            optimizer.zero_grad()
            
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()

        print(f"Epoch {epoch+1} | Loss: {loss_val}")