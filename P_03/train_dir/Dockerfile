
# Usa la imagen base de TensorFlow con soporte para GPU
FROM python:3.9.2

# Actualiza pip
RUN pip install --upgrade pip

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt /requirements.txt

# Instala las dependencias
RUN pip install -r /requirements.txt

# Copia el script de entrenamiento al contenedor
COPY train.py /train.py

# Establece el script de entrenamiento como el comando predeterminado del contenedor
CMD ["python", "/train.py"]
