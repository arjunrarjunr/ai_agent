FROM docker.io/bitnami/python:latest

WORKDIR /root/notebooks

ADD requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 1234
EXPOSE 8501
