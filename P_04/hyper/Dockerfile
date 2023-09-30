
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
FROM gcr.io/deeplearning-platform-release/tf-gpu.2-6.py39

WORKDIR /

# Actualiza pip
RUN pip install --upgrade pip
# Copia el archivo requirements.txt al contenedor
COPY requirements.txt /requirements.txt
# Instala las dependencias
RUN pip install -r /requirements.txt
# Installs hypertune library
RUN pip install cloudml-hypertune

# Copies the trainer code to the Docker image.
COPY . .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "hyper"]
