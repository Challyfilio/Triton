import numpy as np
import matplotlib.pyplot as plt

import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net

from modelz.src.resnet import resnet50, resnet18, resnet152
from callback import EvalCallBack
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    