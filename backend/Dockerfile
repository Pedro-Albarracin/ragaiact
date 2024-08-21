# Usar una imagen base oficial de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo los archivos de requerimientos primero para aprovechar la cache de Docker
COPY requirements.txt .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que uvicorn servirá la aplicación
EXPOSE 8000

# Comando para iniciar uvicorn, sirviendo la aplicación en el puerto 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]