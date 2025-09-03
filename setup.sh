#!/bin/bash

# This script creates a Python virtual environment, activates it,
# and installs the packages listed in requirements.txt.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the virtual environment directory.
VENV_DIR="venv"

# Check if Python3 is available.
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

echo "Creating virtual environment in './${VENV_DIR}'..."
python3 -m venv ${VENV_DIR}

echo "Activating the virtual environment..."
source ${VENV_DIR}/bin/activate

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete! The virtual environment '${VENV_DIR}' is active."
echo "To deactivate it, simply run the command: deactivate"
