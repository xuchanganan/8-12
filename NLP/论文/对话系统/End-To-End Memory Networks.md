&emsp;&emsp;15年的论文, 利用记忆网络做单轮对话.
代码地址: https://github.com/facebook/MemNN.

# Abstract
&emsp;&emsp;该模型是Memory Network的形式, 但是不像Memory Network需要有监督得处理每一层, 它是端到端的训练的, 因此在训练时需要更少的监督. 
&emsp;&emsp;由于模型的灵活性, 可以用作Question-Answering和Language Model中.  

&emsp;&emsp;模型使用了多跳计算(multiple computational hops), 其实就是相同操作层的堆叠, 实验表明该操作可以提升模型结果. 
