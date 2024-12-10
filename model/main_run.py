from training import training
from omegaconf import OmegaConf



import time

from runcontrol import autoencoder
import joblib
import numpy as np
from torch import tensor
import torch


params = OmegaConf.load(r'settings.yaml')
training(params)
