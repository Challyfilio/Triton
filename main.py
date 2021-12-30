'''
可视化验证模型用
Challyfilio
'''
from mindspore_solution import *

if __name__ == '__main__':
    val_data_path = 'data/Tumor/Testing'
    batch_size = 32
    pred_visualize = False

    val_ds = create_dataset(val_data_path, training=False)
    val_ds = val_ds.batch(batch_size=batch_size, drop_remainder=True)

    class_name = {0: "glioma", 1: "meningioma", 2: "no", 3: 'pituitary'}
    image_show(val_ds, class_name)
    
    net = resnet50(class_num=4)
    lr = 0.0005
    # 定义优化器和损失函数
    opt = nn.Adagrad(params=net.trainable_params(), learning_rate=lr, weight_decay=0.05)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')  # 交叉熵

    # 实例化模型
    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})

    # 测试模型用
    net_test(net, 'Luna.ckpt', model, val_ds)
    if batch_size == 32:
        pred_visualize = True
    visualize_model(net,
                    'Luna.ckpt',
                    class_name,
                    val_ds,
                    pred_visualize=pred_visualize)
