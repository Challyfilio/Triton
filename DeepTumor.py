"""
author: Xie RL
"""
from resnet_x import resnet50
from mindspore import nn, Model, context
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, SummaryCollector
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum, Accuracy
import matplotlib.pyplot as plt
import mindspore.common.dtype as mstype
from mindspore.train.callback import Callback


class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["Accuracy"])
            print(acc)


def eval_show(epoch_per_eval):
    plt.xlabel("epoch number")
    plt.ylabel("Model accuracy")
    plt.title("Model accuracy variation chart")
    plt.plot(epoch_per_eval["epoch"], epoch_per_eval["acc"], "red")
    plt.savefig('./acc.png')
    plt.show()


def main():
    train_path = './data/Tumor/Training'
    test_path = './data/Tumor/Testing'
    ckpt_path = './CheckPoint'
    # summary_path = '/root/summary/DeepTumor'

    label_list = {
        'no_tumor': 0,
        'glioma_tumor': 1,
        'meningioma_tumor': 2,
        'pituitary_tumor': 3
    }
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    Training_size = 3265  # 训练集大小
    batch_size = 32  # 每批次大小
    eval_per_epoch = 2  # 检查精度的轮次间隔
    epoch_size = 200  # 伦次数量

    dataset1 = ds.ImageFolderDataset(dataset_dir=train_path, class_indexing=label_list, shuffle=True)
    eval1 = ds.ImageFolderDataset(dataset_dir=test_path, class_indexing=label_list, shuffle=True)

    # 缩放至224*224，并转换为CHW（符合resNet50）
    transforms_list = [c_trans.Decode(),
                       c_trans.Resize(size=[224, 224]),
                       # c_trans.Rescale(1.0 / 255.0, 0.0),
                       c_trans.HWC2CHW()]
    dataset2 = dataset1.map(operations=transforms_list, input_columns=["image"])
    eval2 = eval1.map(operations=transforms_list, input_columns=["image"])

    type_cast_op = C2.TypeCast(mstype.float32)
    dataset3 = dataset2.map(operations=type_cast_op, input_columns=["image"])
    eval3 = eval2.map(operations=type_cast_op, input_columns=["image"])

    # 生成网络
    network = resnet50(class_num=4)

    # loss function ：交叉熵损失
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # optimization definition 优化器：Momentum
    opt = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), learning_rate=0.001, momentum=0.9)

    # 模型确定
    model = Model(network, loss_fn=ls, optimizer=opt, metrics={"Accuracy": Accuracy()})

    # CheckPoint CallBack definition 使用回调函数保存模型
    # save_checkpoint_steps：每执行多少step保存一次（每2个epoch保存一次）
    config_ck = CheckpointConfig(save_checkpoint_steps=int(Training_size / batch_size) * eval_per_epoch,
                                 keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix="train_resnet_Tumor", directory=ckpt_path, config=config_ck)

    # LossMonitor 用于打印loss(每执行多少step打印一次)
    loss_cb = LossMonitor(int(Training_size / batch_size))
    # TimeMonitor 用于打印耗时(每个epoch打印一次)
    Time_cb = TimeMonitor(int(Training_size / batch_size))

    dataset3 = dataset3.batch(batch_size=batch_size, drop_remainder=True)
    eval3 = eval3.batch(batch_size=394, drop_remainder=True)

    # 训练中验证精度
    epoch_per_eval = {"epoch": [], "acc": []}  # 方便最后打印精度曲线
    eval_cb = EvalCallBack(model, eval3, eval_per_epoch, epoch_per_eval)

    # summary_collector = SummaryCollector(summary_dir=summary_path, collect_freq=1)

    model.train(epoch_size, dataset3, callbacks=[ckpoint_cb, loss_cb, Time_cb, eval_cb],
                dataset_sink_mode=False)

    # 绘制精度曲线
    eval_show(epoch_per_eval)


if __name__ == '__main__':
    main()