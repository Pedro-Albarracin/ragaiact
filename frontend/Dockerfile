# Usar una imagen base de Nginx
FROM nginx:alpine

# Copiar los archivos buildados de React al directorio de Nginx
COPY build/ /usr/share/nginx/html

# Copiar la configuración de Nginx para servir en el puerto 3000
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Exponer el puerto 3000 para que coincida con el docker-compose
EXPOSE 3000

# Comando para correr Nginx
CMD ["nginx", "-g", "daemon off;"]