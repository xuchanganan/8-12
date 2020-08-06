# Abstract. 
&emsp;&emsp;我们探索了建立一个又快又好的文本匹配模型什么是必要的，提出了为inter-sequence alignment保留3个关键特征: 1、原始的点对齐特征, 2、先前的点对齐特征和3、上下文特征，同时简化其他特征。  
&emsp;&emsp;在自然语言推理, paraphrase identification(意图识别)和answer selection效果都不错, 而且比其他模型要快6倍。  




# Introduction.
&emsp;&emsp; RE2的架构如图所示, Embedding层首先对离散的tokens进行了嵌入,  -> 一些具有相同结构的block组成了encoder层， 这些blocks通过增强版本的残差连接 -> 接着alignment对齐层 和 fusion融合层 对连续的序列进行了处理 -> 池化层将序列特征 转为一个向量 -> 该向量被用来做最后的预测.  

&emsp;&emsp;
