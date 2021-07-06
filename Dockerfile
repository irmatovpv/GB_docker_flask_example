FROM python:3.8
LABEL maintainer="sten85@mail.ru"
COPY . /app
WORKDIR /app
RUN unzip /app/app/data.zip -d /app/app/
RUN pip install -r requirements.txt
EXPOSE 8180
EXPOSE 8181

COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
