# Abstract 
&emsp;&emsp;强调了一波GPT应用广泛性.

# Introduction.
&emsp;&emsp;从无标签文本数据中利用more than word-level的信息是一项挑战, 主要有两个原因:
- 不知道以什么样的目标, 学习一个怎样的文本表示用于迁移是有效的.
- 将学得的表示用于目标任务没有一个共识.  

&emsp;&emsp;在这篇文章中, 作者提出了解决上述问题的一个尝试: **无监督学习的预训练和有监督的微调相结合**.首先, 我们使用一个语言模型的目标在无标签数据上学习模型的初始参数.然后再用特定任务的目标学习.  
&emsp;&emsp;模型架构上, 使用了Transformer.  
&emsp;&emsp;我们在4类语言理解任务上评估了效果.自然语言推理;question answering;语义相似度;文本分类;  





