# Usa una imagen base de Python 3.10.12
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos
COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip3 install -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Ejecuta las migraciones y luego inicia el servidor con Gunicorn
CMD HOME=/root python3 manage.py runserver 0.0.0.0:8080 --noreload