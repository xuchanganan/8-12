# 2.Our Approach
## 2.1 问题和动机.  
&emsp;&emsp;中文拼写错误纠正(CSC)可以看作是**等长**的序列标注任务: 将input中不正确的字词用正确的替换,从而得到输出. 该任务是比较容易的,然而, 在某种意义上, 仅仅是几个字词需要被替换, 而其他的字词需要直接copy下来。  
&emsp;&emsp;目前做CSC的state-of-art方法是使用Bert完成, 实验表明, **如果指定错误字词, 那么纠错会表现得更好_指Bert-Finetune+Force吧.而基于原始Bert的方法倾向于不做任何修改或者说仅仅只是copy了原始字符. ** 论文作者的解释是:Bert在预训练期间只mask了15%的词, 所以没有学到错误检测.因此需要设计新的模型. 
## 2.2 模型.



## 3.6 消融研究.
- Hard-Masked Bert:  
如果检测网络对当前字符的检测值超过阈值(0.95, 0.9, 0.7), 当前字符的embedding设置为mask的embedding; 否则保留当前字符. 
- Rand-Masked Bert:  
一个字是否错误的概率是随机生成的between 0-1.  
- Bert-Finetune+Force:  
该模型被视为一个上界, 甚至超过论文模型, 该方法中, 让Bert-Finetue仅仅去预测错误的位置.并且从候选集中选择一个字词替换. 
