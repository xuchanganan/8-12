# 清洗英语文本中数据

```
def clean_text(text):
    # 解决缩写问题: 
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    
    text = re.sub(r"\s{2,}", " ", text)   # 将多个空格替换.为一个空格
    # text = re.sub('[^a-zA-Z]',' ', text)  # 匹配目标字符串中非a-z也非A-Z的字符
    
    # 标点符号的匹配. 考虑到情感, 所以保留 ? ! . 标点符号 
    text = re.sub(r"\.", " ", text) # 将. 替换为空格, 比如 a.append -> a append 
    text = re.sub(r"\/", " ", text) # 将/ 替换为空格
    text = re.sub(r"'", " ", text)  # 将' 替换为空格
    text = re.sub(r"[^\w\s?!.]", " ", text)  # 将所有的除了?!.表情感以外的符号都换为 " " 
    
    return text
```
