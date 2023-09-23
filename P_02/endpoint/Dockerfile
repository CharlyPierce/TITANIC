# Usa una imagen base de Python. Puedes ajustar la versión según tus necesidades.
FROM python:3.9-slim

# Establece un directorio de trabajo
WORKDIR /app

# Instala las dependencias necesarias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código del modelo al contenedor
COPY . /app/

# Especifica el comando para ejecutar cuando el contenedor se inicie
CMD ["python", "/app/server.py"]
