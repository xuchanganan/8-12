# Abstract
&emsp;&emsp;对于纠错来说, 现有的一个比较好的方法是:对句子的每个位置利用Bert从候选集中选择一个合适的字来纠错。但是Bert本身是没有足够的能力可以检测到某个位置是否有错, 这是因为pre-training的任务造成的。  

# 2.Our Approach
## 2.1 问题和动机.  
&emsp;&emsp;中文拼写错误纠正(CSC)可以看作是**等长**的序列标注任务: 将input中不正确的字词用正确的替换,从而得到输出. 该任务是比较容易的,然而, 在某种意义上, 仅仅是几个字词需要被替换, 而其他的字词需要直接copy下来。  
&emsp;&emsp;目前做CSC的state-of-art方法是使用Bert完成, 实验表明, **如果指定错误字词,那么纠错会表现得更好_指Bert-Finetune+Force吧,而基于原始Bert的方法倾向于不做任何修改或者说仅仅只是copy了原始字符.论文作者的解释是:Bert在预训练期间只mask了15%的词, 所以没有学到错误检测.因此需要设计新的模型**.
## 2.2 模型.
&emsp;&emsp;检测网络是用了Bi-GRU, 纠错网络是用了BERT.  
&emsp;&emsp;特殊的是, 作者第一次创造了一种embedding方式: 为输入句子中的每个character创建一个embedding作为输入的embedding(our method first creates an embedding for each character in the input sentence, referred to as input embedding).**BERT模型的最后一层包含了一个所有character的softmax function**.
## 2.3 检测网络
## 2.4 纠错网络
&emsp;&emsp;纠错网络是一个基于bert的序列化的多分类标注模型,输入是soft-masked sequence embeddings(e1', e2', ..en')而输出是字序列(y1, y2,...,yn).  
&emsp;&emsp;作者将bert最后一层的hidden sequence定义为Hc = (h1c, h2c, ..., hnc)  
&emsp;&emsp;对于序列中的每个character, 纠错概率被定义为Pc(yi = j|X), 意思是chracter xi被纠正为**候选集中yj的概率**.  
## 2.5 Learning
&emsp;&emsp;BERT是经过预训练的并且训练数据是类似于{(X1, Y1), (X2, Y2),...,(Xn,Yn)}的数据对(bert is pre-trained and training data is given which .. **这个training data是谁的呀.**), 其中X是包含错误的, Y是没有错误的.**一种生成数据的方法是：对于一个Y1, 利用confusion table生成若干个X**.  

# 3.Experimental Results.
## 3.1 Datasets.
&emsp;&emsp;为了确保数据中包含大量的不正确的语句, 特意采样的低质量文本。**作者三个人做了5轮标注:仔细纠正titles中的拼写错误**  
&emsp;&emsp;作者创造了一个困惑词表:每个字与若干个可能混淆的词关联, 随机得替换了15%字去生成错误,其中80%是利用confusion table生成的, 而20%是随机替换的.  

## 3.4 Main Results
- Bert-Pretrain: f1基本只有 4.7  
- Bert-Finetune: f1基本有 71.9 64.9这样.  
**所以推测小米是pretrain.但是pretrain怎么做呢.**
## 3.6 消融研究.
- Hard-Masked Bert:  
如果检测网络对当前字符的检测值超过阈值(0.95, 0.9, 0.7), 当前字符的embedding设置为mask的embedding; 否则保留当前字符. 
- Rand-Masked Bert:  
一个字是否错误的概率是随机生成的between 0-1.  
- Bert-Finetune+Force:  
该模型被视为一个上界, 甚至超过论文模型, 该方法中, 让Bert-Finetue仅仅去预测错误的位置.并且从候选集中选择一个字词替换. 
