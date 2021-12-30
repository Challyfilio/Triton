没有数据集(自行准备)

目录结构及文件说明：

```
./
└─resnet50.ckpt 稍后上传
└─modelz
└─data
   └─Tumor
      └─Training
      │   └─glioma_tumor
      │   └─meningioma_tumor
      │   └─no_tumor
      │   └─pituitary_tumor
      └─Testing
          └─glioma_tumor
          └─meningioma_tumor
          └─no_tumor
          └─pituitary_tumor
└─pytorch_solution.py pytorch框架 不完善
└─mindspore_solution.py mindspore框架
└─callback.py mindspore_solution的回调类
└─main.py 验证模型用(可视化)
└─image_operate.ipynb 图像增广用(OpenCV)
```
扩充数据集图片来源：
- [BrainWeb: Simulated Brain Database](https://brainweb.bic.mni.mcgill.ca/)
- [Kaggle脑肿瘤数据集](https://pan.baidu.com/s/12RTIv-RqEZwYCm27Im2Djw%C2%A0) tave