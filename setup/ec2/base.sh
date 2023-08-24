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

# Start the mlflow server
mkdir mlflow-server
cd mlflow-server
poetry init -n
poetry add mlflow
poetry add boto3
source .venv/bin/activate
nohup mlflow server -h 0.0.0.0 --default-artifact-root s3://receipt-extraction-models &
deactivate
cd ..

# Clone the repository
git clone https://github.com/Cloud-Tech-AI/receipt-data-extraction.git
cd receipt-data-extraction/preprocess
poetry init -n
poetry add boto3
poetry add opencv-python
poetry add shapely
poetry add jellyfish
poetry add pillow
poetry add tqdm
poetry add python-dotenv
