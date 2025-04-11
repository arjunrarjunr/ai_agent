FROM docker.io/bitnami/python:latest

WORKDIR /root/notebooks

ADD requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Install Ollama
#RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose ports for Jupyter Lab,Ollama and Streamlit
EXPOSE 1234
#EXPOSE 11434  
EXPOSE 8501   