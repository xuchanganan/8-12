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
