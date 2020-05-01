# Abstract

# Bert
&emsp;&emsp;Bert包含两个步骤: `pre-training` and `fine-tuning`.  
&emsp;&emsp;在`pre-training`过程中, 模型是针对不同预训练任务、训练在无标签数据上的.  
&emsp;&emsp;在`fine-tuning`过程中, Bert模型初始参数来自预训练模型,针对特定的下游任务,所有的参数通过使用标签数据来微调.不同的下游任务有不同的微调模型,即便它们的初始化参数是相同的. 

&emsp;&emsp;Bert一个独特的特征是:它在不同任务上的统一架构,在`pre-trained architecture`和`downstream architecture`上只有很小的区别. 

## Model Architecture
Bert模型是一个多层双向的Transformer encoder.  
记L为层数; H为hidden size; A为自注意力的头数.  
Bert_Base是为了和GPT做比较弄的, 两者参数量相同,区别在于Bert使用的是双向的self-attention, 而GPT使用的是constrained self-attention(每个token只能看到它的左边,好像类似于Transformers decoder)

