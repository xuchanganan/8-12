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

# 使用建议:
- 初始学习率是非常重要的一个参数（不论是BERT还是其他模型），需要根据目标任务进行调整。
- ERNIE的最佳学习率和BERT/BERT-wwm相差较大，所以使用ERNIE时请务必调整学习率（基于以上实验结果，ERNIE需要的初始学习率较高）。
- **由于BERT/BERT-wwm使用了维基百科数据进行训练，故它们对正式文本建模较好；而ERNIE使用了额外的百度贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。**
- 在长文本建模任务上，例如阅读理解、文档分类，BERT和BERT-wwm的效果较好。
- 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
- 如果要处理繁体中文数据，请使用BERT或者BERT-wwm。因为我们发现ERNIE的词表中几乎没有繁体中文。

**这里提到的ERNIE特指百度提出的ERNIE**
