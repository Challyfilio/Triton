目录结构及文件说明：

```
./
├─resnet50.ckpt	            # 预训练模型 ImageNet2012
├─BestModel.ckpt            # 最终结果
├─modelz                    # resnet模型文件夹
├─data
│  └─Tumor
│     ├─Training
│     │   ├─glioma_tumor
│     │   ├─meningioma_tumor
│     │   ├─no_tumor
│     │   └─pituitary_tumor
│     └─Testing
│         ├─glioma_tumor
│         ├─meningioma_tumor
│         ├─no_tumor
│         └─pituitary_tumor
├─pytorch_solution.py       # pytorch框架_不完善
├─mindspore_solution.py     # mindspore框架
├─callback.py               # mindspore_solution的回调类
├─main.py                   # 验证模型用(可视化)
└─image_operate.ipynb       # 图像增广用(OpenCV)
```
相关文件下载：
- [数据集、预训练模型](https://pan.baidu.com/s/10ej677Z8Se3kGUdEQpc7Xw) 提取码：3721

扩充数据集图片来源：
- [BrainWeb: Simulated Brain Database](https://brainweb.bic.mni.mcgill.ca/)
- [Kaggle脑肿瘤数据集](https://pan.baidu.com/s/12RTIv-RqEZwYCm27Im2Djw%C2%A0) 提取码：tave