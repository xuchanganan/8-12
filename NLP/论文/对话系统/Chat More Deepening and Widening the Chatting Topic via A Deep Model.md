# ABSTRACT
&emsp;&emsp;论文研究的是**开放领域的**，通过**生成回复**来做的**多轮对话系统**, 论文的目的是为了拓宽，深入聊天主题。为此使用了3个channel:global channel, wide channel, deep channel.  
&emsp;&emsp;**global channel**编码了完整的历史信息  
&emsp;&emsp;**wide channel**使用了一个基于attention机制的循环神经网络来预测没有出现在历史信息中的keywords关键词.  
&emsp;&emsp;**deep channel**使用了多层感知机选择一些keywords来in-depth discussion.  
&emsp;&emsp;最后模型融合了3个channel的output去生成合适的responses.  

# Introduction

&emsp;&emsp;目前主要有以下几点难点.  
- 在长文本中, 无关的话语往往比相关的多得多, 如何识别出relevant words去引导回复是个未解决的问题.  
- 不去加深或拓宽某个主题的聊天是无聊的, 因此生成一些不仅相关, 而且话题上更深的回复是个难题.  
- 大规模数据集  

&emsp;&emsp;为了解决上面的问题, 论文作者提出了DAWnet模型.  
- 首先DAWnet将对话划段  
- 全局的channel首先将给定的上下文encoder为一个embedding vector,该向量包含了所有的历史对话信息.  
- DAWnet从context中提取了全部的关键词,**训练时关键词其实是事先提取好的, 属于监督学习, 但预测的时候没有.**, 在此关键词的基础上, wide channel使用了一个基于attention机制的RNN网络去预测更wider的关键词. 更广泛的关键词意味着可能没有在given context中出现, 并且可以帮助扩宽主题.  
- deep channel使用了MLP去选择一些更深的关键词为了更深层次的讨论, 因此MLP的输入是context embedding vector和之前得到的关键词.  
- 最后使用contextual encoder的输出、在deep channel中提前选择的keywords、和wide channel中预测的keywords, 然后做一个decoder来生成一个有意义的回复.  

# 3.Model
&emsp;&emsp;模型首先将对话分为多段, 并且从历史信息中提取出一些关键词.  
&emsp;&emsp;然后模型将context和它的关键词喂入3个平行channels  
- global 编码context为一个embedding vector.  
- wide 预测更广的关键词.  
- deep channel 基于context和它的keywords选择更深的keywords.  
&emsp;&emsp;最后模型使用了一个attention机制去加权context以及keywords，然后将它们喂入RNN decoder中用来生成response.  

## 3.1 Keyword Extraction
&emsp;&emsp;使用的是TF-IDF指标去提取一个context中的keywords.  
&emsp;&emsp;论文作者移除了停用词, 仅仅保留了DailyDialog和新浪微博对话库中的nouns, verbs, 和adjectives.  
&emsp;&emsp;将一个session对话看作是一个document, 将一个单词看作是一个term然后去计算每个单词的TF-IDF值.最后从每个session中选择top 20的关键词  

## 3.2 Global Channel
&emsp;&emsp;利用多层RNN将给定的上下文信息编码为一个vector.  

## 3.3 Wide Channel
&emsp;&emsp;该channel训练了一个基于attention的rnn模型去预测extend topics的关键词, 对于给定向量c, 用于关键词预测的RNN以encoder的最后一个hidden state初始化, 并且根据下述方式进行更新.  
st = f(st-1, [ekt-1p, ct]), ekt-1p是在t-1时刻预测关键词的embedding.ct是t时刻通过attention机制得到的vector, [ekt-1p, ct]是两个向量的拼接,st是RNN在时刻t的隐层状态.
&emsp;&emsp;公式解释见Notability.  
&emsp;&emsp;在这个channel下, decoder实际得预测了一些更广的关键词, 这些关键词可以被喂入decoder中用作回复生成.

## 3.4 Deep Channel.
&emsp;&emsp;目标是从context中**选择一些有用的keywords**去加深聊天话题, 通过用一个带RELU激活函数的MLP模型去计算keywords的权重, 输入是encoder最后一个hidden state以及所有keywords的embedding vectors.输出由下列公式给出.  
&emsp;&emsp;l0 = [ht, ek1c, ek2c, ..., ekmc]
&emsp;&emsp;q = MLP(l0)

&emsp;&emsp;选择的keywords的向量通过mi = qi* Wt * ekic来更新,这部分选择的keywords将会被送入decoder中去生成回复.

## 3.5 Decoder.
&emsp;&emsp;用于生成回复的decoder RNN和用于生成keyword的decoder RNN其实是相似的, 但是vector向量c是不同的.  
&emsp;&emsp;公式详见Notability.  

