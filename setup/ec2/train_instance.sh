#!bin/bash

# Update the system
sudo apt update
sudo apt upgrade -y

# Install General Dependencies
sudo apt install git -y
sudo dpkg -S /usr/bin/nohup

# Install Python 3.11
sudo apt install python3.11 -y
sudo apt install python3-pip -y
sudo pip install --upgrade pip

# Install the required Python packages
sudo pip install poetry
poetry config virtualenvs.in-project true

# Clone the repository
git clone https://github.com/Cloud-Tech-AI/receipt-data-extraction.git
cd receipt-data-extraction
poetry install
