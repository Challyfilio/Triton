# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from mindspore import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # lr = nn.dynamic_lr.natural_exp_decay_lr(learning_rate=0.05,
    #                                         decay_rate=0.5,
    #                                         total_step=200,
    #                                         step_per_epoch=10,
    #                                         decay_epoch=10)
    # print(lr)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
