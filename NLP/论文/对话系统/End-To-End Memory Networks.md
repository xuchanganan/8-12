&emsp;&emsp;15年的论文, 利用记忆网络做单轮对话.  
&emsp;&emsp;代码地址: https://github.com/facebook/MemNN.

# Abstract
&emsp;&emsp;该模型是Memory Network的形式, 但是不像Memory Network需要有监督得处理每一层,由于原始的Memory Network无法反向传播. 但该论文是端到端的训练的, 所以在训练时需要更少的监督.  
&emsp;&emsp;由于模型的灵活性, 可以用作Question-Answering和Language Model中.  

&emsp;&emsp;模型使用了多跳计算(multiple computational hops), **其实就是相同操作层的堆叠, 实验表明该操作可以提升模型结果**.

# Introduction
&emsp;&emsp;最近出现了很多使用 显示存储 和 attention 的网络结构.  
&emsp;&emsp;该论文中, 作者提出了一个类似RNN的网络架构:在输出一个symbol之前, 多次从一个尽可能大的存储单元读取.  

&emsp;&emsp;作者认为在长期记忆过程中, 多次读取(multiple hops)对于模型效果是有很大帮助的.  

# Approach

