# Abstract

# Bert
&emsp;&emsp;Bert包含两个步骤: `pre-training` and `fine-tuning`.  
&emsp;&emsp;在`pre-training`过程中, 模型是针对不同预训练任务、训练在无标签数据上的.  
&emsp;&emsp;在`fine-tuning`过程中, Bert模型初始参数来自预训练模型,针对特定的下游任务,所有的参数通过使用标签数据来微调.不同的下游任务有不同的微调模型,即便它们的初始化参数是相同的. 

&emsp;&emsp;Bert一个独特的特征是:它在不同任务上的统一架构,在`pre-trained architecture`和`downstream architecture`上只有很小的区别. 

### Model Architecture
&emsp;&emsp;Bert模型是一个多层双向的Transformer encoder.  
&emsp;&emsp;记L为层数; H为hidden size; A为自注意力的头数.  
&emsp;&emsp;Bert_Base是为了和GPT做比较弄的, 两者参数量相同,区别在于Bert使用的是双向的self-attention, 而GPT使用的是constrained self-attention(每个token只能看到它的左边,好像类似于Transformers decoder)  
  
### Input/Output Representations
&emsp;&emsp;为了使Bert可以处理大量的下游任务, 我们的input需要清楚得区分出来`single sentence`和`sentences pair`在一个`token sequence`中.  
**这里的sentence可以是任意的连续文档, 而不是一个实际的linguistic sentence**.  
**这里的sequence指#的是Bert的input token sequence, 可以是一句sentence,也可以是sentences pair**.  

&emsp;&emsp;使用的是30000个token词表的WordPiece embeddings. 每个sequence的第一个token是CLS, CLS在最后的隐层中对应的是用于分类任务的sequence向量.  
&emsp;&emsp;语句对被打包在一个sequence中(即input token sequence).  

**我们通过以下两种方式区分input token sequence中的两个sentences:**  
- 通过特殊分隔符SEP来分隔
- 为每个token增加了一个embedding(应该是segment embedding)来标记token是属于sentence A还是sentence B.  

对于每个给定的token, 它的input representation = tokenEmbedding + segmentEmbedding + positionEmbedding. 

## Pre-training BERT  
&emsp;&emsp;我们没有使用传统的left-to-right或者right-to-left语言模型去预训练Bert,而是使用了两个无监督任务:`Masked LM`和`Next Sentence Prediction(NSP)`

### Masked LM  
&emsp;&emsp;不幸的是, 传统的语言模型只能left-to-right 或者 right-to-left训练, **待补充**  
为了训练一个`deep bidirectional representation`,我们简单得随机mask inputs tokens, 然后去预测`masked token`. 在这种情况下, 对应masked tokens的隐层向量被输入到softmax中去预测.  

实验中, 在每个sequence(即一个样本)中随机mask 15%的WordPiece tokens, 不同于denoising auto-encoders, 我们仅仅预测masked words而不是重新预测整个输入.  

**这样做虽然可以获得一个bidirectional pre-trained model, 但是缺点是:预训练和微调的不匹配(微调任务中的样本是没有mask的)**  
下面是解决方法:  
**为了解决这个问题,并没有每次都mask掉word, 而是随机选择15%的token positions, 然后对选中的15%以80%的概率拿mask去替换, 以10%的概率不变, 以10%的概率去拿句子中的其他词替换. 然后, 该词对应的隐向量将会被用来预测原始的token, 使用的损失函数是交叉损失熵**.  
详见C2.  
**通过实验得知:fine-tuning对不同的masking strategies有鲁棒性, 这里的mask百分比不包括0%和100%**.

### Next Sentence Prediction(NSP)  
&emsp;&emsp;为了训练一个可以理解句子间关系的模型, 谷歌预训练了一个"是否是下一个句子"的二值预测任务,该任务可以轻松从任何一个单语语料库中生成.特殊地,需要保证:B是A的下一句话的概率是50%,**输出C被用来判断B是否是A的下一句话**  

待译,黄色部分未理解:  
`in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters.`  
### Pre-training data
&emsp;&emsp;预训练程序和现存的语言模型预训练基本一致.**它是很重要的,去使用一个文档级别的语料库,而不是一个打乱的语句级别的语料库,使用文档级别的语料库是为了提取长的连续的语句对.**  

## Fine-tunning Bert
&emsp;&emsp;不同于传统Transformers处理句子对(encoder+decoder), Bert将这个过程结合在了一起(所以只用到了Transformer-encoder部分),将句子对拼接在一起直接送入encoder中了.  
对于每个任务,只需要简单将input和output送入bert即可,然后端到端微调参数.  

Bert中的SentenceA和SentenceB可以类比于以下几种:  
- sentence pairs in paraphrasing
- hypothesis-premise pairs in entailment
- 问答系统中的问题-段落对
- 非生成问题text-null,比如文本分类和序列标注  
  
&emsp;&emsp;**在输出中,token 向量被喂到输出层做token级别的任务，比如序列标注和问答; CLS向量被喂到输出层用来做分类, 比如情感分析.**  
