## 使用方法

* 直接在main中切换train(spark)与test(spark)即可进行训练与预测模式
* 保存模型与加载模型的路径需要调整，注意，这是一个**目录**，不是**文件**
* 预测结果的路径也需要调整，同样的，这是一个**目录**
* 将预测结果目录下两个part-00000,-00001之类的csv结果直接拼接在一起，再对0/1**取个反**即可提交结果。如果提交得到0分，可能是没有取反
* 想要换模型的话，直接在train中的model=xxx()与test中的model=xxx()中的xxx改为所需要的模型即可