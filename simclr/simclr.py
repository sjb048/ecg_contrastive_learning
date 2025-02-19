# IMPORTS
import  os
import  random
import  numpy as np
import  pandas as pd
import pywt
import  time

import  torch
import  torch.nn as nn
from    torchvision import transforms

from network import Simple1DCNN
from utils import accuracy

model = Simple1DCNN(in_channels=1, num_features=128)

class SimCLR(object):
    def __init__(self):
        pass