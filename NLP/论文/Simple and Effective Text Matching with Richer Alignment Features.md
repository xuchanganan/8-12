# Abstract. 
&emsp;&emsp;我们探索了建立一个又快又好的文本匹配模型什么是必要的，提出了为inter-sequence alignment保留3个关键特征: 1、原始的点对齐特征, 2、先前的点对齐特征和3、上下文特征，同时简化其他特征。  
&emsp;&emsp;在自然语言推理, paraphrase identification(意图识别)和answer selection效果都不错, 而且比其他模型要快6倍。  




# Introduction.
&emsp;&emsp; RE2的架构如图所示, Embedding层首先对离散的tokens进行了嵌入,  -> 一些具有相同结构的block：由encoder、alignment和fusion层组成, 这些blocks间通过增强版本的残差连接 -> 池化层将序列特征 转为一个向量 -> 该向量被用来做最后的预测.  

&emsp;&emsp;

# Our Approach.
&emsp;&emsp;两个文本序列在预测层之前是被对称处理的, 并且除了预测层的参数外, 对于两个文本序列来说, 其他层的参数都是共享的. **Figure 1省略了另一条序列的处理流程.**  
&emsp;&emsp; Encoder中的N个相同结构的blocks是有着各自独立的参数，并且通过增强的残差连接. **在每个block中, 序列encoder首先计算了序列的上下文特征(图中的实体黑框)**.Encoder的输入和输出之后被concat在一起, 然后喂入对齐层去学习两个序列间的alignment和interaction. 融合层融合了对齐层的输入和输出. **融合层的输出可以看作是一个block的输出**, 最后一个block的输出被送到池化层中并且转换为一个固定长度的向量, **预测层拿两个向量作为输入并且预测了最终的target**.交叉损失被用来优化分类任务中的损失.  

&emsp;&emsp;每一层的实现都是尽可能简单, **在embedding仅仅使用了word embeddings, 没有用到字向量和其他语法信息**. **在encoder中使用了多层的卷积网络(vanilla multi-layer convolutional networks with same padding)**. **在池化层使用了 max-over-time池化操作**.  

## 1.增强的残差连接.  
&emsp;&emsp;对于长度为l的序列, 我们定义第n个block的输入为x^(n)=(x1^(n), x2^(n), ..., xl^(n))输出为o^(n)=(o1^(n), o2^(n), ..., ol^(n)), 特殊地让o^(0)作为一个全0的序列向量, 而第一个block的输入x^(1)是embedding层的输出. 对于第n层block(n>=2)的输入是x^(1)和前两个blocks的输出和的concat, 即:xi^(n)=[xi^(1); oi^(n-1) + oi^(n-2)], ;代表concat操作.  
&emsp;&emsp;**因此,对齐层和融合层的输入有3部分: embedding vectors(空白矩阵), residual vectors就是之前block的输出(对角线矩阵), encoded vectors通过encoder后的输出(黑色矩阵)**.  

## 2.对齐层.  
&emsp;&emsp;计算形式来源于attention, 对齐层将两个seqences作为输入, 并且将计算的结果作为输出. 第一个长度为la的sequence input被标记为a = (a1, a2, .., ala), 第二个长度为lb的sequence input被标记为b = (b1, b2, ..., blb).a, b的相似度得分eij 被下列公式计算所得: eij = F(ai)^T * F(bj). F是一个函数或者是一个单层的前向传播.  
&emsp;&emsp;输出向量a'和b'是另一个序列的加权和计算所得, 详情见论文公式3.  

[seq_len_a, hidden_size]  [seq_len_b, hidden_size]  ->  [seq_len_a, seq_len_b]  矩阵维度是 [hidden_size, hidden_size]
如果不等长呢?
源码中是有mask的。

## 3.融合层.   
&emsp;&emsp;融合层在三个视野下比较了local representations和aligned representations，并且将其融合在了一起, 具体见公式4.  
&emsp;&emsp; G1，G2，G3和G是单层的前向传播网络,使用的是独立的参数, o 表示的是点乘, -强调了两个向量的不同, 而o 强调了两个vector之间的相似度.  

## 4.预测层.
&emsp;&emsp;预测层将池化层的输出v1和v2作为输入, 并且预测了最后的结果.  
**预测层也提供了3种选择_供选择**, H是一个多层的前向传播网络.  

# 3.实验.

## 3.2实现细节. 
&emsp;&emsp;用NLTK对句子分词, 全部小写, 移除所有标点, 没有限制最大的seq_length, 一个batch中所有的seq 被 padding到这个batch最长的那个长度. word_embedding是用840B-300d的GLOVE初始化的, 而且训练中不变, 没有出现的单词embedding被初始化为0, 也固定不变. 每个全连接层或卷积层前都有一个drop out. 预测层是2层的神经网络. 前向反馈网络激活函数都是gelu. blocks是1-3个, 卷积层也是1-3个.


# 存在疑问. 
- 为什么是[xi^(1); oi^(n-1) + oi^(n-2)]  
这里的o(n-1)和o(n-2)为什么是直接相加呢?  
**因为是残差连接呀！！___自己理解好像可以减少方差, 所以说+不仅仅是因为不同属性的波**  


# 已解决. 
- 为什么Alignment输出的是基于另一个句子的向量表示呢?  
因为Fusion融合层的输入是当前句子的向量表示+另一个句子的向量表示，这样才有交互融合呀。

- 拼接会保留序列关系，但...总觉得怪怪的。
