import argparse
import os
import sys
import random
import yaml
import logging
import logging.handlers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime as dt


def setup_config_args(parser, filepath: str = 'config.yaml'):
    """Load a YAML configuration"""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)[parser.parse_args().dataset]
    f.close()

    """Setup argparse based on the YAML configuration"""
    for key, value in config.items():
        arg_type = type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'{key} (default: {value})')
    return parser.parse_args()


def fix_seed(seed: int = 2024):
    """Fix random seed everywhere"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    return

def get_device(cuda_id):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id  # Set the GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device

def get_logger(filepath: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fileHandler = logging.FileHandler(filepath)
    streamHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'
    )
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


def make_route(dir_path, file_name=None):
    # Full path for the directory
    absolute_path = os.path.join(os.getcwd(), dir_path)

    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # only for making directory
    if file_name is None:
        return

    # Full path for the file inside the directory
    file_path = os.path.join(absolute_path, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        # Get the current date and time
        current_datetime = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Define the new filename with the current date and time
        title, extension = os.path.splitext(file_name)
        new_file_name = f'{title}-backup-{current_datetime}-{extension}'
        # Rename the existing file
        new_file_path = os.path.join(absolute_path,new_file_name)
        os.rename(file_path, new_file_path)

    # Create a new file (or open the file if it somehow already exists) and write something to it
    f = open(file_path, 'w')
    f.close()

    return

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': nn.PReLU(),
        'rrelu': F.rrelu,
        'celu': nn.CELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU()
    }

    return activations[name]

def save_heatmap(matrix, title, xlabel, ylabel, save_path, clim_min=None, clim_max=None, _dpi=300, _facecolor="#eeeeee",
                 _bbox_inches='tight'):
    plt.clf()
    plt.matshow(matrix)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    if clim_min != None and clim_max != None:
        plt.clim(clim_min, clim_max)
    plt.savefig(save_path, dpi=_dpi, facecolor=_facecolor, bbox_inches=_bbox_inches)

    return

def save_np(dir_path, file_name, np_data):
    if os.path.exists(dir_path):
        print("directory already exists")
    else:
        os.mkdir(dir_path)
    np.save(dir_path+'/'+file_name, np_data)
    print("{} is saved successfully".format(file_name))
    return
