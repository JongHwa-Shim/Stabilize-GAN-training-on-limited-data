import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime

writer = SummaryWriter('./tblogs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

def tblog(step, **kwargs):
    for key, value in kwargs:
        writer.add_scalar(key, value, step)

#tensorboard --logdir=./tblogs/