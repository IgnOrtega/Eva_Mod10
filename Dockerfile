# Imagen base
FROM python:3.12

# Directorio de trabajo
WORKDIR /app

# Crear el directorio 'results' en el contenedor
RUN mkdir -p /app/serializar_modelo

# Copiar archivos, incluyendo el modelo desde la carpeta 'results'
COPY serializar_modelo/Modelo_Breast_Cancer.pkl /app/serializar_modelo/
COPY app.py .
COPY requirements.txt .
# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 5248

# Ejecutar app
CMD ["python", "app.py"]

#- Para crear el contenedor
# docker build -t breast-cancer-api .