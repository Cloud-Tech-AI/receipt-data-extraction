#!bin/bash

# Update the system
sudo apt update
sudo apt upgrade -y

# Install AWS CLI v2
curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install the CodeDeploy agent
sudo apt install -y ruby
sudo apt install -y wget
wget https://aws-codedeploy-us-west-2.s3.us-west-2.amazonaws.com/latest/install
chmod +x ./install
sudo ./install auto

# Clean up temporary files
rm -rf awscliv2.zip aws install

# Install General Dependencies
sudo apt install git -y
sudo dpkg -S /usr/bin/nohup

# Install Python 3.11
sudo apt install python3.11 -y
sudo apt install python3-pip -y
pip install --upgrade pip

# Install the required Python packages
pip install poetry
poetry config virtualenvs.in-project true

# Start the mlflow server
mkdir mlflow-server
cd mlflow-server
poetry init -n
poetry add mlflow
poetry add boto3
source .venv/bin/activate
nohup mlflow server -h 0.0.0.0 --backend-store-uri postgresql://ishan_mlflow:mlflow_db@localhost:5432/mlflow_db --default-artifact-root s3://receipt-extraction-models &
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
poetry add psycopg2-binary
