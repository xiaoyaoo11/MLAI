#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if [ "$1" == "--install" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Run the Flask application
echo "Starting web server..."
flask run --host=0.0.0.0 --port=5000
