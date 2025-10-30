import subprocess, shlex

flags = ['dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'dermamnist-e', 'breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e']
colors = [False, True, False, True, True, True, False, False, False, True, False, True, True, False, True]  # Colors for the flags
#batch_sizes = [32, 640, 128, 128, 640, 640, 640, 640, 128]  # Batch sizes for the flags

use_randaugments = [False, False, False, False, False, True, True, True, True, True, True, True, True, True]         # <- enable/disable RandAugment here
num_epochs = 100
batch_size = 128
python = "/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12"  # or path to your venv python

for f, c, r in zip(flags, colors, use_randaugments) :
    cmd = f"{python} /mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/train_resnet18_medMNIST.py --flag {shlex.quote(f)} --color {str(c)} --batch_size {str(batch_size)} --use_randaugment {str(r)} --num_epochs {str(num_epochs)} --cuda cuda:2"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)