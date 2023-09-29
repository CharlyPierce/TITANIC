
# Usa la imagen base de TensorFlow con soporte para GPU
#FROM python:3.9.2
#Para una imagen con CPU, Python 3.9, y Debian 11:
#FROM gcr.io/deeplearning-platform-release/tf-2-6-cpu-v20230822-debian-11-py39
# Para una imagen con GPU (CUDA 11.3), Python 3.9, y Debian 11:
FROM gcr.io/deeplearning-platform-release/tf-gpu.2-6.py39
# Para una imagen con TPU, Python 3.9, y Debian 11:
#FROM gcr.io/deeplearning-platform-release/tf-2-6-tpu-v20230822-debian-11-py39


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
