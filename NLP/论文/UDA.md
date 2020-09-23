# ABSTRACT
&emsp;&emsp;一般的半监督学习：在大量的无标注数据上, 使用协同训练去约束模型的预测，以起到对噪声的鲁棒性。  
&emsp;&emsp;而作者的工作主要是:如何有效得往五标注数据上添加噪声，而且认为高质量的噪声，尤其是用先进的数据扩增方法生成的噪声样本在半监督学习中起着很重要的角色。  
&emsp;&emsp;通过使用先进的数据扩增方法，例如:RandAugment和BackTranslation来代替普通的噪声操作，模型在nlp和cv上有着很大的提升。  
&emsp;&emsp;该方法在迁移学习上同样有效: 比如Bert.

# Introducion
&emsp;&emsp;半监督学习SSL是利用无标注数据去解决深度学习需要大量标注难题的很好的办法，目前主要是基于consistency training的方式，而且这种方式在很多基准上有不错的效果  
&emsp;&emsp;通俗来讲,consistency training method简单得对模型和预测结构进行了正则化，以保证面对input examples noise和hidden states noise时，模型具有鲁棒性，该framework之所以有意义是因为:一个好的模型应该对input example或hidden states的微小改动有鲁棒性，在这个framework下，不同模型的差异性主要体现在噪声注入位置和噪声注入方式的不同上.目前典型的噪声注入方式是：高斯噪声、dropout noise或者是对抗噪声.  
&emsp;&emsp;在该份工作中，作者探索了consisency training中噪声注入所扮演的角色, 并且调研了先进的数据扩增方法：具体来说，在监督学习中效果好的，在半监督学习中效果一样也好，这说明在监督学习中数据扩展的表现和他们在consistency training中的表现有着很强的相关性。因此，作者建议使用先进的数据扩增方式去替换consistency training中的传统的噪声注入方法.  
&emsp;&emsp;作者发现:当标签数据很大的时候, UDA同样有效: 比如ImageNet, 当使用全部标签数据和1.3M的无标签数据时，acc from 78.43 to 79.05.

作者认为贡献主要如下：  
- 说明了：在监督学习中好的数据扩展方法，在consistency enforcing半监督学习中可以充当更高级的噪声来源  
- UDA可以匹配，甚至超过更多数据的监督学习  
- UDA可以和迁移学习很好地迁移，比如BERT  
- 针对UDA为何能提高分类效果？数据扩增起什么样的角色？给出了解释。

# 2.Unsupervised data augmentation(UDA)  
