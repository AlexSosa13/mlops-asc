# Imagen base oficial de Python, versión slim para reducir tamaño
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# IMPORTANTE: Instalar curl para que el HEALTHCHECK pueda ejecutarse
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias ANTES que el código fuente
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Puerto que expone el contenedor
EXPOSE 8000

# 1. Configuración del Health Check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Bind en 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]