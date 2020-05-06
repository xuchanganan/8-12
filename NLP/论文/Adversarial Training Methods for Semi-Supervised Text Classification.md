[原作代码链接](https://github.com/tensorflow/models/tree/e97e22dfcde0805379ffa25526a53835f887a860/research/adversarial_text)  
# Abstract
Adversarial training提供了一种正则化监督学习算法, 而**Virtual adversarial training则有能力将监督学习扩展到半监督学习(不大懂)**.然而,**这两种方法都是在input vector上加的扰动，
这样是不合适的,尤其是当输入input vector是稀疏高维的特征表示, 比如说one-hot编码**.  
**论文选择了在word embeddings上添加扰动,而不是原始的输入input本身**.  
该方法在multiple benchmark半监督任务和完全的监督任务中都取得了不错的结果.在可视化结果中可以看到**习得的word embeddings在质量上有提升,而且在训练的时候,模型也不大会倾向于过拟合**.  


