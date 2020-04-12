# 参考资料
[Top 8 方案](https://zhuanlan.zhihu.com/p/101554661)  
- Pseudo-Lable(伪标签)
- Hyper Parameters(初始学习率设置)
- TTA
- EDA  
它说：EDA效果不明显, 甚至无效果.   

[Top 1 方案](https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION)
- 数据预处理  
相比传统模型, 基于bert的深度模型往往不需要数据预处理就可以取得很好的效果。在本赛题，**数据预处理效果不稳定，但可以用在模型融合增加差异上。**
- 数据不平衡  
上采样（下采样）直接改变数据分布; 训练中, 调整不同类别样本的权重. 效果不明显
- Loss的设计  
对于F1指标, 将cross_entropy换成了**可求导的f1_loss**, 提升效果不稳定. 
- BERT模型优化  
除了cls输出, 也对内部隐含层输出进行提取, 拼接成更全面的特征向量 .
- 伪标签  
预测结果加入训练集, **在训练过程引入测试集分布。** 
- 模型融合  
**改变模型结构和输入数据制造差异性**

[kaggple Top1](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557)
