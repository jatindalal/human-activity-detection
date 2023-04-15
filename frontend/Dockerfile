FROM nginx:1.15.8-alpine

RUN apk update && apk add openssl vim

WORKDIR /etc/nginx/certs/
RUN openssl req -x509 \
            -sha256 -days 356 \
            -nodes \
            -newkey rsa:2048 \
            -subj "/CN=demo.mlopshub.com/C=US/L=San Fransisco" \
            -keyout rootCA.key -out rootCA.crt 

COPY ./nginx.conf /etc/nginx/nginx.conf

COPY ./*.html /usr/share/nginx/html/
COPY ./*.css /usr/share/nginx/html/
COPY ./*.js /usr/share/nginx/html/

EXPOSE 80
EXPOSE 443
