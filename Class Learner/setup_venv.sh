#!/bin/bash

echo "Setting up virtual environment for the Class Learner project..."

python -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
