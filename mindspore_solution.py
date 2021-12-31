"""
v4.0
2021/12/30 还有一天就跨年啦！
Challyfilio
"""
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

from modelz.src.resnet import *
from modelz.src.CrossEntropySmooth import CrossEntropySmooth
from callback import EvalCallBack
from sklearn.metrics import accuracy_score, classification_report


def create_dataset(data_path, training=True):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    # 对数据进行增强操作
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if training:
        trans = [
            CV.Decode(),
            CV.Resize(size=[224, 224]),
            # CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.0)),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.RandomVerticalFlip(prob=0.1),
            CV.RandomColorAdjust(contrast=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Decode(),
            CV.Resize(size=[256, 256]),
            CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    type_cast_op = C.TypeCast(mstype.float32)

    # 实现数据的map映射、批量处理和数据重复的操作
    data_set = data_set.map(operations=trans, input_columns='image', num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns='image', num_parallel_workers=8)
    return data_set


def image_show(ds, class_name, count=1):
    data = next(ds.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    labels = Tensor(labels, mstype.int32)
    print("Tensor of image", images.shape)
    print(images.dtype)
    print("Labels:", labels)
    print(labels.dtype)

    # 输出测试图
    plt.figure(figsize=(12, 7))
    for i in images:
        plt.subplot(4, 8, count)
        picture_show = np.transpose(i.asnumpy(), (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean

        picture_show = picture_show / np.amax(picture_show)
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.title(class_name[int(labels[count - 1].asnumpy())])
        plt.xticks([])
        count += 1
        plt.axis("off")
    plt.show()


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


# 模型验证
def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def curve_draw(record):
    sns.set_theme(style="whitegrid")
    loss_value = record['loss']
    loss_value = list(map(float, loss_value))
    plt.plot(record['epoch'], record['acc'], label='acc')
    plt.plot(record['epoch'], loss_value, label='loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    # plt.savefig('./acc.png')


# 验证方法
def net_test(net, best_ckpt_path, model, ds):
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    acc = model.eval(ds)
    print("\n{}".format(acc))


# 定义网络并加载参数，对验证集进行预测
def visualize_model(net, best_ckpt_path, class_name, val_ds, pred_visualize=False):
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss, metrics={"Accuracy": nn.Accuracy()})
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    # pred和labels 计算准确率
    print('\nAccuracy is: ' + str(accuracy_score(pred, labels)) + '\n')
    print(classification_report(pred, labels))

    if pred_visualize:
        # 可视化模型预测
        plt.figure(figsize=(12, 7))
        for i in range(len(labels)):
            plt.subplot(4, 8, i + 1)
            color = 'blue' if pred[i] == labels[i] else 'red'
            plt.title(class_name[pred[i]], color=color)
            picture_show = np.transpose(images[i], (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            picture_show = std * picture_show + mean

            picture_show = picture_show / np.amax(picture_show)
            picture_show = np.clip(picture_show, 0, 1)
            plt.imshow(picture_show)
            plt.axis('off')
        plt.show()
    else:
        pass


if __name__ == '__main__':
    # GPU
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    train_data_path = 'data7/Tumor/Training'
    val_data_path = 'data7/Tumor/Testing'

    train_ds = create_dataset(train_data_path, training=True)
    val_ds = create_dataset(val_data_path, training=False)

    class_name = {0: "glioma", 1: "meningioma", 2: "no", 3: 'pituitary'}
    net = resnet50(class_num=4)
    batch_size = 32
    num_epochs = 200

    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
    val_ds = val_ds.batch(batch_size=394, drop_remainder=True)
    image_show(train_ds, class_name)
    # image_show(val_ds, class_name)

    # 加载预训练模型
    # pretrained = 'Luna.ckpt'
    pretrained = 'Triton.ckpt'
    param_dict = load_checkpoint(pretrained)

    # 获取全连接层的名字
    filter_list = [x.name for x in net.end_point.get_parameters()]

    # 删除预训练模型的全连接层
    filter_checkpoint_parameter_by_list(param_dict, filter_list)

    # 给网络加载参数
    load_param_into_net(net, param_dict)

    # ——————————————
    # 冻结除最后一层外的所有参数
    # for param in net.get_parameters():
    #     if param.name not in ["end_point.weight", "end_point.bias"]:
    #         param.requires_grad = False
    # ——————————————

    lr = 0.0005
    # 定义优化器和损失函数
    # opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)
    # opt = nn.Adam(params=net.trainable_params(), learning_rate=lr)
    opt = nn.Adagrad(params=net.trainable_params(), learning_rate=lr, weight_decay=0.05)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # 交叉熵
    # loss = CrossEntropySmooth(sparse=True, reduction='mean',
    #                           smooth_factor=0.2,
    #                           num_classes=4)

    # 实例化模型
    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})

    eval_param_dict = {"model": model, "dataset": val_ds, "metrics_name": "Accuracy"}
    epoch_per_eval = {"epoch": [], "loss": [], "acc": []}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, epoch_per_eval, )

    # 训练模型
    print('-' * 20)
    print(pretrained + '\nepoch=' + str(num_epochs) + '\nbatch=' + str(batch_size) + '\n')
    model.train(num_epochs,
                train_ds,
                callbacks=[eval_cb, TimeMonitor()],
                dataset_sink_mode=False)

    # print(epoch_per_eval)
    curve_draw(epoch_per_eval)

    net_test(net, 'best.ckpt', model, val_ds)
    visualize_model(net, 'best.ckpt', class_name, val_ds, pred_visualize=False)
