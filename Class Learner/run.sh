#!/bin/bash

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

if [ "$1" == "--install" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo "Starting web server..."
flask run --host=0.0.0.0 --port=5000
