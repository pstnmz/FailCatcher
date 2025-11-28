import subprocess, shlex

flags = ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e']
colors = [False, False, False, True, False, True, True, False, True]  # Colors for the flags

use_randaugments = [False, False, False, False, False, False, False, False, False]  # <- enable/disable RandAugment here
use_dropout = False        # <- enable/disable Dropout for MC Dropout
dropout_rate = 0.1         # <- dropout rate (default: 0.1 for ViT, lower than ResNet)
learning_rate = 0.0001     # <- learning rate (ViT typically uses lower LR)
num_epochs = 100
batch_size = 128            # <- ViT uses smaller batch size due to larger model

python = "/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12"  # or path to your venv python

for f, c, r in zip(flags, colors, use_randaugments):
    cmd = f"{python} /mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/trainings/train_vit_medMNIST.py --flag {shlex.quote(f)} --color {str(c)} --batch_size {str(batch_size)} --use_randaugment {str(r)} --use_dropout {str(use_dropout)} --dropout_rate {str(dropout_rate)} --learning_rate {str(learning_rate)} --num_epochs {str(num_epochs)} --cuda cuda:2"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)
