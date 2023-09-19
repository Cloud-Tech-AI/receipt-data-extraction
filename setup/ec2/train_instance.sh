#!bin/bash

# Update the system
sudo apt update
sudo apt upgrade -y

# Install General Dependencies
sudo apt install git -y
sudo dpkg -S /usr/bin/nohup

# Install Pip
sudo apt install python3-pip -y
sudo pip install --upgrade pip

# Install the required Python packages
sudo pip install poetry
poetry config virtualenvs.in-project true

# Clone the repository
git clone https://github.com/Cloud-Tech-AI/receipt-data-extraction.git
cd receipt-data-extraction
poetry install
poetry shell

# Install PyTorch (CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'



############################################################################################################

# NVIDIA DRIVERS AND TOOLKIT
# PyTorch (2.0.1+cu118) comes with its own CUDA runtime, but it does not include the NVIDIA drivers or CUDA toolkit. The NVIDIA drivers are required to communicate with the GPU, and the CUDA toolkit is required to compile and run CUDA code.

############################################################################################################

# Install NVIDIA drivers (nvidia-driver-535 + CUDA 12.2--->latest)
sudo apt autoremove nvidia* --purge
sudo apt install nvidia-driver-535 -y
reboot
nvidia-smi

# Install CUDA toolkit (CUDA 11.8) Not Needed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
