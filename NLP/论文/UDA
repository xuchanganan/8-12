# ABSTRACT
&emsp;&emsp;一般的半监督学习：在大量的无标注数据上, 使用协同训练去约束模型的预测，以起到对噪声的鲁棒性。  
&emsp;&emsp;而作者的工作主要是:如何有效得往五标注数据上添加噪声，而且认为高质量的噪声，尤其是用先进的数据扩增方法生成的噪声样本在半监督学习中起着很重要的角色。  
&emsp;&emsp;通过使用先进的数据扩增方法，例如:RandAugment和BackTranslation来代替普通的噪声操作，模型在nlp和cv上有着很大的提升。  
&emsp;&emsp;该方法在迁移学习上同样有效: 比如Bert.

# Introducion
&emsp;&emsp;半监督学习SSL是利用无标注数据去解决深度学习需要大量标注难题的很好的办法，目前主要是基于consistency training的方式，而且这种方式在很多基准上有不错的效果  
&emsp;&emsp;通俗来讲,consistency training method简单得对模型和预测进行了正则化，以保证面对input examples noise和hidden states noise时，模型具有鲁棒性，在这个framework下，模型的差异性主要体现在噪声注入位置和噪声注入方式的不同上。
