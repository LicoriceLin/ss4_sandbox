cd SS4
az storage blob download-batch -d . -s traindata --pattern AF_SWISS/*  --account-name reseek0db
az storage blob download --account-name reseek0db --container-name traindata --name AF_SWISS_64 -f  AF_SWISS_64.tar.gz
az storage blob upload --account-name reseek0db --container-name traindata --name reseek_AF_SIWSS-v2.tar.gz  --file reseek_AF_SIWSS-v2.tar.gz --account-key JsaGCOgCZ06HQx7/jma7jSDkYd5j/3mfNhg5IW2QKS7LdPRXZ6pQxfKzc9qgAA0CIaX6PF5pmBo1+AStY9noYQ==
az storage blob download --account-name reseek0db --container-name traindata  --name reseek_AF_SIWSS-v2.tar.gz  --file reseek_AF_SIWSS-v2.tar.gz --account-key JsaGCOgCZ06HQx7/jma7jSDkYd5j/3mfNhg5IW2QKS7LdPRXZ6pQxfKzc9qgAA0CIaX6PF5pmBo1+AStY9noYQ==
git clone https://github.com/LicoriceLin/ss4_sandbox.git