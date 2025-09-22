#!/bin/bash

# Start the Ollama server in the background
echo "Starting Ollama server to preload models: $MODEL_NAMES"
ollama serve &

# Capture the PID of the Ollama server
OLLAMA_PID=$!

# Wait for the server to be ready (adjust if necessary)
echo "Waiting for Ollama server to start..."
sleep 5

# Split the comma-separated model names into an array
IFS=',' read -r -a MODELS <<< "$MODEL_NAMES"

# Loop through each model and pull it
for MODEL_NAME in "${MODELS[@]}"; do
  echo "Pulling model: $MODEL_NAME"
  if ollama pull "$MODEL_NAME"; then
    echo "Successfully pulled model: $MODEL_NAME"
  else
    echo "Failed to pull model: $MODEL_NAME"
    kill $OLLAMA_PID
    exit 1
  fi
done

# Stop the Ollama server
echo "Stopping Ollama server..."
kill $OLLAMA_PID

# Wait for the server to terminate
wait $OLLAMA_PID

echo "Model preloading complete."