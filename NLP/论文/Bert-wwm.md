# 基本信息  
[官方github链接](https://github.com/ymcui/Chinese-BERT-wwm)  
Pre-Training with Whole Word Masking for Chinese BERT. (基于whole word masking技术的中文预训练模型BERT-wwm)  

# 全词mask和部分mask的区别.  

原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。 在全词Mask中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即全词Mask.
需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成[MASK]标签的情况.

由于谷歌官方发布的BERT-base, Chinese中，中文是以字为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。 我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了哈工大LTP作为分词工具，即对组成同一个词的汉字全部进行Mask。
|说明|样例|
| -- | -- |
| 原始文本 | 使用语言模型来预测下一个词的probability |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] |  
