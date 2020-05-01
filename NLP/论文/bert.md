# Abstract

# Bert
&emsp;&emsp;Bert包含两个步骤: `pre-training` and `fine-tuning`.  
在`pre-training`过程中, 模型是针对不同预训练任务、训练在无标签数据上的.  
在`fine-tuning`过程中, Bert模型初始参数来自预训练模型,针对特定的下游任务,所有的参数通过使用标签数据来微调.不同的下游任务有不同的微调模型,即便它们的初始化参数是相同的.  


