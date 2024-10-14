wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh -b
conda init
source /home/azureuser/.bashrc
conda activate base

pip3 install torch torchvision torchaudio
conda install mamba ipykernel -c conda-forge
pip install lightning datasets torchmetrics transformers "lightning[pytorch-extra]" #"jsonargparse[signatures]"
pip install numpy pandas scipy matplotlib seaborn

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
conda install gh -c conda-forge

az login
gh auth login
git config --global user.name 'Zhenfeng Evan Deng'
git config --global user.email 'zhenfengdeng121@gmail.com'

pip cache purge
conda clean --all

