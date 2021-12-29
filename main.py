# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from mindspore import nn

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    milestone = [2, 5, 100]
    learning_rates = [0.1, 0.05, 0.01]
    lr = nn.dynamic_lr.natural_exp_decay_lr(learning_rate=0.05,
                                            decay_rate=0.5,
                                            total_step=200,
                                            step_per_epoch=10,
                                            decay_epoch=10)
    print(lr)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
