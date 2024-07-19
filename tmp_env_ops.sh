pip3 install torch torchvision torchaudio
conda install mamba ipykernel -c conda-forge
pip install lightning
pip install torchmetrics transformers "lightning[pytorch-extra]" #"jsonargparse[signatures]"
pip install numpy pandas scipy matplotlib seaborn

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
conda install gh -c conda-forge

az login
gh auth login
git config --global user.name 'Zhenfeng Evan Deng'
git config --global user.email 'zhenfengdeng121@gmail.com'

pip cache purge
conda clean --all

