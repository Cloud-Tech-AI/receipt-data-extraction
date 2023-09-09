#!bin/bash

# Update the system
sudo apt update
sudo apt upgrade -y

# Install General Dependencies
sudo apt install git -y
sudo dpkg -S /usr/bin/nohup

# Upgrade pip
pip install --upgrade pip
pip install --upgrade setuptools

# Install the required Python packages
pip install poetry
poetry config virtualenvs.in-project true

# Clone the repository
git clone https://github.com/Cloud-Tech-AI/receipt-data-extraction.git
cd receipt-data-extraction
poetry install
