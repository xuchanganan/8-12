# 结构
与LSTM相比,GRU有几个主要差异:  
- GRU将两个状态(单元状态和最终隐藏状态)组合成单个隐藏状态ht, 这样减少了参数数量.  
- 取消了输出门,引入了一个**复位门**, 当它接近1时, 在计算当前状态时完全读取先前状态信息。当复位门接近0时, 则在计算当前状态时忽略先前的状态.  
- GRU将输入门和遗忘门组合成一个**更新门**, 如果更新门是0, 则将先前单元状态的全部状态信息送入当前单元状态.如果更新门是1,则将所有当前输入送入当前单元状态。
