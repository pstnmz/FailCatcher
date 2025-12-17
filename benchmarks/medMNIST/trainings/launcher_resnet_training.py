import subprocess, shlex
from pathlib import Path

#flags = ['organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e', 'breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e']
#colors = [False, False, False, True, False, True, True, False, True, False, False, False, True, False, True, True, False, True]  # Colors for the flags
#batch_sizes = [32, 640, 128, 128, 640, 640, 640, 640, 128]  # Batch sizes for the flags
flags=['organamnist', 'organamnist', 'organamnist', 'organamnist']
colors=[False, False, False, False]  # Colors for the flags
#use_randaugments = [False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True]         # <- enable/disable RandAugment here
use_randaugments = [False, True, False, True]         # <- enable/disable RandAugment here
use_dropouts = [False, False, True, True]        # <- enable/disable Dropout for MC Dropout
#use_dropouts = True        # <- enable/disable Dropout for MC Dropout
dropout_rate = 0.3         # <- dropout rate (default: 0.3)
num_epochs = 100
batch_size = 128

python = "/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12"  # or path to your venv python
script_path = Path(__file__).parent / 'train_resnet18_medMNIST.py'

for f, c, r, d in zip(flags, colors, use_randaugments, use_dropouts) :
    cmd = f"{python} {script_path} --flag {shlex.quote(f)} --color {str(c)} --batch_size {str(batch_size)} --use_randaugment {str(r)} --use_dropout {str(d)} --dropout_rate {str(dropout_rate)} --num_epochs {str(num_epochs)} --cuda cuda:2"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)