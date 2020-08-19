# ABSTRACT
&emsp;&emsp;短文本匹配在信息检索、问答和对话系统中扮演着重要角色, 文章设计出Multi-Channel Information Crossing(MIX)，一个可以用于文本匹配生产环境的多通道卷积神经网络，附带基于句子和语义特征的注意力机制。
MIX在不同粒度上比较了不同的文本片段，并形成一系列多通道相似矩阵，然后与另一个精心设计的注意力矩阵交互，将丰富的句子结构暴露给深度神经网络, **多亏精心设计的多通道信息交互，使得模型有很大提升。**

# 1.Introduction
&emsp;&emsp;早期的文本匹配方法包括:基于知识库检索的自动问答以及基于词匹配和特征交互的特殊检索， 随后就是基于CNN——RNN等的深度学习网络模型，**但是实践看来，这些模型投入到生产中，仍然有很大的提升空间，尤其是当深度模型和语言结构和语义特征结合起来。**
&emsp;&emsp;MIX是一个在不同粒度上的注意力机制的融合体, 主要的想法如下所示:一、MIX提取了很多不同粒度的信息，术语，短语，语法和语义，术语词频，权重甚至是语法信息，作者观察到文本匹配在不同粒度的特征结合将会最大化深度模型的能力：express all levels of local dependences并且在卷积时最小化信息损失。二、MIX也提出了新的fusion技术将从不同通道获取的匹配结果结合起来，这里有两种类型的通道，通过这两种通道,两个文本片段的特征可以交互。**第一种类型是语义信息通道，**这个通道包含了文本的意思例如一元组、二元组和三元组。**第二种通道类型包含了结构化信息，**比如：术语权重，Part-Of-Speech和命名实体还有一些交互的特征空间。在MIX中, **语义信息通道在相似度匹配上有用，而结构化信息通道主要用于注意力机制**,而且，**MIX使用3D卷积核去处理这些堆叠的层，从多通道中提取抽象特征并且通过一个多层感知机器结合输出**。**这个通道结合机制，允许MIX轻松得添加新的通道，以使得MIX应用在大量的任务中**  

# 2. Preliminaries and background
&emsp;&emsp;取决于转换和匹配的顺序, 文本匹配模型可以被分为两类：基于表示Presentation的和基于交互Interaction的.**基于表示的是先将两个文本片段用深度神经网络转为tensor得到向量化表示，比如：DSSM，CDSSM和CNTN等，然后在两句话的向量表示上做匹配**。而后者**基于交互的则是先对于每个文本对生成一个交互矩阵，然后利用神经网络去提取有用的特征，并且从交互矩阵中去学习有用的匹配模式，例如：Arc-I, MatchPyramid和DRMM**  

# 3，MIX MODEL



WikiQA is a publicly accessible dataset containing open-domain
question and answer pairs provided by Microsoft.

NDCG@3, which is a popular metric for measuring the ranking quality and is widely adopted in search engine
evaluation.
