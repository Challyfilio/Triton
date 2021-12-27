"""learning rate generator"""
import numpy as np


def get_lr(current_step, lr_max, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       current_step(int): current steps of the training
       lr_max(float): max learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
