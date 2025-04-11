#!/bin/bash
# filepath: c:\Users\MrRajaramParthasarat\Desktop\Project\python\entrypoint.sh

# Start Ollama server in the background
ollama serve &

# Wait for the Ollama server to be ready
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/models > /dev/null; do
  sleep 2
done

# Pull the llama2 model
echo "Pulling llama2 model..."
ollama pull llama2

# Start Jupyter Lab in the background
echo "Starting Jupyter Lab..."
jupyter lab --no-browser --ip=0.0.0.0 --port=1234 --allow-root &

# Start Streamlit app
echo "Starting Streamlit app..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0