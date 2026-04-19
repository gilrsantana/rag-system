#!/bin/bash
# Entrypoint script to pull required Ollama models

set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready (using bash built-in, curl not available)
echo "Waiting for Ollama to start..."
until bash -c 'exec 3<>/dev/tcp/localhost/11434' 2>/dev/null; do
    sleep 2
done
echo "Ollama is ready!"

# Pull required models
echo "Pulling required models..."
echo "Pulling nomic-embed-text:latest (embedding model)..."
ollama pull nomic-embed-text:latest

echo "Pulling llama3.2:3b (base LLM)..."
ollama pull llama3.2:3b

echo "All models pulled successfully!"

# Keep Ollama running in foreground
wait
