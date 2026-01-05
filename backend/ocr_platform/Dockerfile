# Usar uma imagem base leve do Python 3.13
FROM python:3.13-slim

# Instalar dependências de sistema necessárias para o OpenCV (cv2)
# O "libgl1" e "libglib2.0" são cruciais para processamento de imagem
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Definir a pasta de trabalho dentro do contentor
WORKDIR /app

# Copiar os requisitos primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Instalar as dependências Python
# Adicionamos --no-cache-dir para manter a imagem leve
RUN pip install --no-cache-dir -r requirements.txt

# Instalar opencv-python explicitamente para garantir
RUN pip install opencv-python

# Copiar todo o código do projeto para dentro do contentor
COPY . .

# Criar pastas necessárias para persistência
RUN mkdir -p uploads segments models data

# Comando padrão (será substituído pelo docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]