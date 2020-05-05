# 论文全名
Focal Loss for Dense Object Detection 其实就是针对类别不平衡提出的损失函数.  

# Abstract  
&emsp;&emsp;解决的是目标检测中,正负样本极度不平衡问题, 目标太少, 背景太多.  
&emsp;&emsp;**论文提出通过重新构造标准交叉损失熵函数去减少分类效果较好的样本损失.让训练集中在一些较少的、难分类的样本上**,并且防止容易分类的分类有压倒性的优势.    
&emsp;&emsp;该论文此外还提出了RetinaNet,目标是训练速度快,而且准确率也高.  

# Focal Loss 
#### 一般CE损失的问题?  
- &emsp;&emsp;正如论文中蓝色曲线所示: **这个损失的一个很大的特点是: 即使是容易分类的类别(p >> 0.5)也会带来一些不小的损失, 这样在容易分类的样本很多的时候, 虽然每一个的损失都比较小, 但是容易分类的样本变多, 损失就会变大. 会overwhelm较少类别的大损失**。
#### 3.1 Balanced Cross Entropy. 
&emsp;&emsp;解决类别不平衡问题的寻常方法是: 为class 1引入一个alpha [0, 1], 而class 0则为1-alpha.  
&emsp;&emsp;这里为了和Pt的写法一致,对写法进行了改写.  

#### 3.2 Focal Loss Definition. 
&emsp;&emsp;**容易分类的负类占据了loss的主体,并且主导了gradient. alpha虽然平衡了positive/negative的重要性(这里说的是类别吧),但是并没有解决easy/hard样本间的不平衡.**  
&emsp;&emsp;因此, Focal Loss在交叉损失熵中引入了可调的因子(1-pt)^gamma, gamma>=0.  
&emsp;&emsp;论文提及Focal Loss的两个性质:  
- 样本如果是错误分类的,并且pt也很小, 那么调节因子是接近1的,它的loss几乎是不受影响的.如果pt->1,那么调节因子是接近于0的, 易分类样本的loss会被减小.  
- 而gammer smoothly adjusts the rate at which easy examples are down-weighted, 也就是说gammer 控制了易分类样本loss的下降程度. 如果gamma=0,那么FL和CE是等价的, 实验发现gamma=2在论文实验中效果最好.  
**we note that the implementation of the loss layer combines the sigmoid operation for computing p with the loss computation, resulting in greater numerical stability.(不大懂)**  

&emsp;&emsp;尽管在论文实验中, 使用的是上面的公式形式,但是实验证明这个形式并不固定,附录中的其他代替公式也同样有效。  

#### 3.3 Class Imbalance and Model Intialization.  
&emsp;&emsp;**默认情况下, 模型初始时, 分类模型正类和负类是相同的概率.但是这样在训练初期,样本多的那类会占据主要的损失导致训练的不稳定**.  
&emsp;&emsp;为了解决这个问题, 引入了一个concept of a 'prior' for the value of p estimated by the model for the rate class(foreground) at the start of training. 可能是说:当少的类别概率>0.01就被判为正确吧. **这里不是很懂.待细看5.1及源码.**

#### 3.4 Class Imbalance and Two-stage Detectors.
