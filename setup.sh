#!/bin/bash

# DeepSeek-Coder RAG Application Setup Script
echo "Setting up DeepSeek-Coder RAG Application..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/indices
mkdir -p models

# Set up environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit the .env file with your configuration details!"
else
    echo ".env file already exists, skipping..."
fi

# Download embedding model for faster first use
echo "Pre-downloading sentence transformer model for embeddings..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Setup complete! Now you can run the application with:"
echo "- For Streamlit web UI: python main.py streamlit"
echo "- For Telegram bot: python main.py telegram"
echo ""
echo "Note: Make sure to edit the .env file with your Telegram bot token if using the Telegram interface."