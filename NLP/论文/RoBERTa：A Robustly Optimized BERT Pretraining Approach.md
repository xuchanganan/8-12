# Abstract  
&emsp;&emsp;语言模型预训练,不同的参数选择在最后的结果上有着明显的影响.  
&emsp;&emsp;作者发现Bert是明显的欠训练的(undertrained), 并且完全可以匹配甚至超过在它之后发布的一些模型.  
These results highlight the importance of previously overlooked design choices.

# RoBERTa  
我们在BERT预训练程序中提出了一些修改(可以提高端任务表现), RoBERTa其实也叫Robustly optimized BERT approach.  
**RoBERTa主要做了以下的改变:**  
- **动态mask**. 
- **Full-Sentences without NSP损失**,也即是说:取消了NSP任务.  
- **更大的mini-batches和一个更大的BPE(byte-level)**  

论文作者也探索了两个其他重要的因素:  
- **用于预训练的数据**.  
- **在数据上跑的轮数**.  
XLNet架构比原始BERT多训练了10倍的数据量, 更大的batch size.同样也多训练了好多epoches.（难道XLNet只是比Bert多训练了那么多吗？）  


从论文Table 4可以看出, 160GB的数据, 8K的batchsize, 500K的steps. 这都没有过拟合哦。

# Training Procedure Analysis  
该部分主要探讨了哪些选择对于成功预训练BERT model是比较重要的.    

#### 4.1 Static vs Dynamic Masking  
- 原始Bert:  
&emsp;&emsp;在预训练过程中,只对输入进行了一次mask。所以称为静态mask.  
- 改进的静态mask:  
&emsp;&emsp;为了避免每个epoch的训练数据都是使用的相同的mask, 训练数据被复制了10次, 这样就有了10份不同的mask,然后在40个epoch中使用了这10个不同的mask, 这样就相当于每个样本在训练的时候被不同的mask，mask了4次。 - 完全的动态mask:
**每个样本输入到model中,都需要重新生成一下mask.这个对大数据集进行更多次的预训练是很重要的**。  

最后的结果..好像SQuAD问答效果明显, 其他两个好像..不大行(可能数据集的问题吧..)  
SST-2情感分析的上了一个十分点.  
但是MNLI-m自然语言推理的下降了3个十分点. 

#### 4.2 模型输入格式和NSP.  
NSP损失可被视为原始Bert模型的一个重要因素.  
有论文指出:移除NSP对模型有损伤,尤其是QNLI,MNLI,和SQuAD  
也有人质疑NSP loss的必要性.  
然后就这个问题, 论文作者做了实验.  

- SEGMENT-PAIR+NSP:  
输入包含两部分，每个部分是来自**同一文档或者不同文档的segment(segment是连续的多个句子),** 两个segment 的token总数少于 512 。预训练包含 MLM 任务和 NSP 任务。这是原始 BERT 的做法。
- SENTENCE-PAIR + NSP：
输入也是包含两部分，每个部分是来自**同一个文档或者不同文档的单个句子，这两个句子的token 总数少于 512 。** 由于这些输入明显少于512 个tokens，因此增加batch size的大小，以使 tokens 总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务。
- FULL-SENTENCES：
输入只有一部分（而不是两部分），来自**同一个文档或者不同文档的连续多个句子**，token 总数不超过 512 。**输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token**。预训练不包含 NSP 任务。
- DOC-SENTENCES：
输入只有一部分（而不是两部分），输入的构造类似于FULL-SENTENCES，只是不需要跨越文档边界，其输入来自同一个文档的连续句子，token 总数不超过 512 。在文档末尾附近采样的输入可以短于 512个tokens， 因此在这些情况下动态增加batch size大小以达到与 FULL-SENTENCES 相同的tokens总数。预训练不包含 NSP 任务。
pass... 
#### 4.3 Training with large batches  
**过去在神经网络机器翻译的工作也表明了:以一个非常大的mini-batches训练可以提高优化速度和端任务的表现, 前提是学习率也被正确的增加. 最近的工作也表明BERT也有这样的特性**  

batchsize = 256, steps = 1M 与 batchsize = 2K, steps = 125K 在计算资源上是等价的, 但是效果却是batchsize要好.  
论文作者发现,在大的batches上面训练可以提高masked language modeling的困惑度, 以及端任务的准确率.  

#### 4.4 Text Encoding  
&emsp;&emsp;Byte-Pair Encoding(BPE)是字词级别的混合, 它可以处理在自然语言语料库上的大量常见词汇, **BPE不是完整的单词，而是依赖于子单词单元，这些子单词单元是通过对训练语料库进行统计分析来提取的**  


# Introduce
这里主要讲的是RoBERTa较历史研究做了哪些改进，以及BERT的一些问题.  
pass 
- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。(**好奇：如何弄更大的byte?**)
