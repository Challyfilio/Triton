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


def create_dataset(data_path, batch_size=24, repeat_num=1, training=True):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    # 对数据进行增强操作
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if training:
        trans = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
            # CV.Decode(),
            # CV.Resize(size=[224, 224]),
            # CV.Rescale(1.0 / 255.0, 0.0),
            # CV.HWC2CHW()
        ]
    else:
        trans = [
            # CV.Decode(),
            CV.Resize(256),
            CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    type_cast_op = C.TypeCast(mstype.float32)  #######

    # 实现数据的map映射、批量处理和数据重复的操作
    data_set = data_set.map(operations=trans, input_columns='image', num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns='image', num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


# 模型验证
def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


# 定义网络并加载参数，对验证集进行预测
def visualize_model(best_ckpt_path, model, val_ds):
    net = resnet50(class_num=4)
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    acc = model.eval(val_ds)  ########
    print("\nnew{}".format(acc))
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss, metrics={"Accuracy": nn.Accuracy()})
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    class_name = {0: "glioma", 1: "meningioma", 2: "no", 3: 'pituitary'}
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    '''
    to-do
    pred和labels 计算准确率
    print('\nAccuracy is: '+str(accuracy_score(pred,labels))+'\n')
    print(classification_report(pred,labels))
    '''

    # 可视化模型预测
    plt.figure(figsize=(12, 5))
    for i in range(len(labels)):
        plt.subplot(3, 8, i + 1)
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


def curve_draw(record):
    plt.xlabel('Epoch')
    loss_value = record['loss']
    loss_value = list(map(float, loss_value))
    plt.plot(record['epoch'], loss_value, 'red', label='loss')
    plt.plot(record['epoch'], record['acc'], label='acc')
    plt.legend()
    plt.show()
    # plt.savefig('./acc.png')


def test_net(network, model, ds_eval):
    """定义验证的方法"""
    # acc = model.eval(ds_eval, dataset_sink_mode=False)
    acc = model.eval(ds_eval)
    print("\n{}".format(acc))


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


if __name__ == '__main__':
    # GPU
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    train_data_path = 'data/Tumor/Training'
    val_data_path = 'data/Tumor/Testing'

    train_ds = create_dataset(train_data_path)

    data = next(train_ds.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    labels = Tensor(labels, mstype.int32)
    print("Tensor of image", images.shape)
    print(images.dtype)
    print("Labels:", labels)
    print(labels.dtype)

    class_name = {0: "glioma", 1: "meningioma", 2: "no", 3: 'pituitary'}
    count = 1

    # 输出测试图
    plt.figure(figsize=(12, 5))
    for i in images:
        plt.subplot(3, 8, count)
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

    net = resnet50(class_num=4)
    num_epochs = 2

    # 加载预训练模型
    param_dict = load_checkpoint('resnet50.ckpt')

    # 获取全连接层的名字
    filter_list = [x.name for x in net.end_point.get_parameters()]

    # 删除预训练模型的全连接层
    filter_checkpoint_parameter_by_list(param_dict, filter_list)

    # 给网络加载参数
    load_param_into_net(net, param_dict)

    # 定义优化器和损失函数
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 实例化模型
    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})

    # train_ds = create_dataset(train_data_path)
    # print(train_ds)
    val_ds = create_dataset(val_data_path)
    eval_param_dict = {"model": model, "dataset": val_ds, "metrics_name": "Accuracy"}
    epoch_per_eval = {"epoch": [], "loss": [], "acc": []}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, epoch_per_eval, )

    # 训练模型
    model.train(num_epochs,
                train_ds,
                callbacks=[eval_cb, TimeMonitor()],
                dataset_sink_mode=True)

    test_net(net, model, val_ds)

    # print(epoch_per_eval)
    curve_draw(epoch_per_eval)

    visualize_model('best.ckpt', model, val_ds)
