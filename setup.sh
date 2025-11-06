#!/bin/bash

echo "================================================"
echo "Document Q&A System - Setup Script"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✓ Python 3 is installed: $(python3 --version)"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠ Ollama is not installed."
    echo "Please install Ollama from: https://ollama.ai"
    echo "Then run: ollama pull llama3"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is running"
    else
        echo "⚠ Ollama is not running. Please start it with: ollama serve"
    fi
fi

echo ""
echo "Setting up virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Verifying .env file..."
if [ ! -f ".env" ]; then
    echo "⚠ .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please edit it and add your LangSmith API key."
else
    echo "✓ .env file exists"
fi

echo ""
echo "Testing configuration..."
python src/config.py

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your LangSmith API key (optional)"
echo "2. Make sure Ollama is running: ollama serve"
echo "3. Start the application: streamlit run app.py"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
