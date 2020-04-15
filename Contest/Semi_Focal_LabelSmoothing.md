# 半监督学习
伪标签的权重应该比正确标签的小, 这里伪标签取了0.5, 训练集取了1.0 

```
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import os
from transformers import *


def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids, input_masks, input_segments = return_id(instance, 'longest_first', max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        ids, masks, segments = _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)
            ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int) + 1)


class WeiboBERT(tf.keras.Model):
    def __init__(self):
        super(WeiboBERT, self).__init__(name='Weibo_bert')
        config = BertConfig.from_pretrained(BERT_PATH + 'bert-base-chinese-config.json', output_hidden_states=True)
        self.bert_model = TFBertModel.from_pretrained(BERT_PATH + 'bert-base-chinese-tf_model.h5', config=config,
                                                      name='bert')

        self.concat = tf.keras.layers.Concatenate(axis=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.15)
        self.output_ = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        input_id, input_mask, input_atn = inputs
        sequence_output, pooler_output, hidden_states = self.bert_model(input_id, attention_mask=input_mask,
                                                                        token_type_ids=input_atn)
        h12 = tf.reshape(hidden_states[-1][:, 0], (-1, 1, 768))
        h11 = tf.reshape(hidden_states[-2][:, 0], (-1, 1, 768))
        h10 = tf.reshape(hidden_states[-3][:, 0], (-1, 1, 768))
        h09 = tf.reshape(hidden_states[-4][:, 0], (-1, 1, 768))
        concat_hidden = self.concat(([h12, h11, h10, h09]))
        x = self.avgpool(concat_hidden)
        x = self.dropout(x)
        x = self.output_(x)  # 这里用了 一层全连接层.
        return x


def Focal_Loss_label_smoothing(y_true, y_pred, alpha=0.5, gamma=2, smoothing=0.001):
    semi_weight = y_true[:, -1]
    y_true = y_true[:, :-1]
    y_true = (1 - smoothing) * y_true + smoothing * (1 / 3)  # 标签平滑.    这里的3 = y_true.shape[-1], 可是此处不能使用.
    y_pred += tf.keras.backend.epsilon()

    ce = -y_true * tf.math.log(y_pred)  # 交叉损失熵.
    weight = tf.pow(1 - y_pred, gamma)  # 降低正确类, 容易分类的权重, 其他的也该略微小点...

    fl = ce * weight * alpha
    reduce_fl = tf.reduce_sum(fl, axis=-1)  # 交叉做了累加.

    reduce_fl = tf.reduce_mean(reduce_fl * semi_weight, axis=-1)
    return reduce_fl



TRAIN_PATH = '/data/changan/train_dataset/'
TEST_PATH = '/data/changan/test_dataset/'
BERT_PATH = '/data/changan/bert_base_chinese/'
dummy_PATH = '/data/changan/train_dataset/'
MAX_SEQUENCE_LENGTH = 140
input_categories = '微博中文内容'
output_categories = '情感倾向'


df_train = pd.read_csv(TRAIN_PATH + 'nCoV_100k_train.labled.csv')
df_train = df_train[df_train[output_categories].isin(['-1', '0', '1'])]
df_train[input_categories] = df_train[input_categories].fillna(value="无")

df_test = pd.read_csv(TEST_PATH + 'nCov_10k_test.csv')
df_test[input_categories] = df_test[input_categories].fillna(value="无")
# 导入伪标签样本.
df_dummy = pd.read_csv(dummy_PATH + 'dummy_dataset.csv')
df_dummy[input_categories] = df_dummy[input_categories].fillna(value="无")

# 待重新预测的. 
df_unmask = pd.read_csv(TRAIN_PATH + 'nCoV_900k_train.unlabled.csv')[:500000]
df_unmask[input_categories] = df_unmask[input_categories].fillna(value="无")


tokenizer = BertTokenizer.from_pretrained(BERT_PATH + 'bert-base-chinese-vocab.txt')
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = compute_output_arrays(df_train, output_categories)
dummy_inputs = compute_input_arrays(df_dummy, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
dummy_outputs = compute_output_arrays(df_dummy, output_categories)

test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
unmask_inputs = compute_input_arrays(df_unmask, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# 开始训练. 
gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99).split(X=df_train[input_categories].fillna('-1'),
                                                                       y=df_train[output_categories].fillna('-1'))
all_train_idx = []
all_valid_idx = []
valid_preds = []
valid_labels = []  # 这里也加了.
test_preds = []
unmask_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    all_train_idx.append(train_idx)
    all_valid_idx.append(valid_idx)
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]

    # 在outputs 弄 weight .
    label_list = [[1.0] for _ in range(len(train_idx))]
    label_list = np.array(label_list)
    train_outputs = to_categorical(outputs[train_idx])
    train_outputs = np.concatenate((train_outputs, label_list), axis=1)

    dummy_list = [[0.5] for _ in range(len(dummy_outputs))]
    dummy_list = np.array(dummy_list)
    # 未标注样本, 弄全部的.
    onehot_dummy_outputs = to_categorical(dummy_outputs)
    onehot_dummy_outputs = np.concatenate((onehot_dummy_outputs, dummy_list), axis=1)

    # 合在一起.
    train_inputs = [np.concatenate((inputs[i][train_idx], dummy_inputs[i]), axis=0) for i in range(len(inputs))]
    train_outputs = np.concatenate((train_outputs, onehot_dummy_outputs), axis=0)

    # print(train_inputs[0].shape)
    # print(train_outputs.shape)

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    label_list = [[1.0] for _ in range(len(valid_idx))]
    label_list = np.array(label_list)
    valid_outputs = to_categorical(outputs[valid_idx])
    valid_outputs = np.concatenate((valid_outputs, label_list), axis=1)

    # 构建模型.
    model = WeiboBERT()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # 数据应该打乱下.
    state = np.random.get_state()
    np.random.shuffle(train_inputs[0])
    np.random.set_state(state)
    np.random.shuffle(train_inputs[1])
    np.random.set_state(state)
    np.random.shuffle(train_inputs[2])
    np.random.set_state(state)
    np.random.shuffle(train_outputs)

    # 损失函数.
    FL = lambda y_true, y_pred: Focal_Loss_label_smoothing(y_true, y_pred, alpha=0.25, gamma=2, smoothing=0.001)

    model.compile(loss=FL, optimizer=optimizer, metrics=['acc'])
    # 验证集也要弄loss的.
    model.fit(train_inputs, train_outputs, validation_data=[valid_inputs, valid_outputs], epochs=1, batch_size=32)

    valid_preds.append(model.predict(valid_inputs))
    valid_labels.append(outputs[valid_idx])  # 这里加了.
    test_preds.append(model.predict(test_inputs))
    unmask_preds.append(model.predict(unmask_inputs))
    K.clear_session()

    
# 存储五折交叉验证划分集合
np.save('/output/all_train_idx.npy', all_train_idx)
np.save('/output/all_valid_idx.npy', all_valid_idx)
# 存储验证集
np.save('/output/valid_preds.npy', valid_preds)
np.save('/output/valid_labels.npy', valid_labels)
# 存储测试集
np.save('/output/test_preds.npy', test_preds)
# 存储未标注数据. preds
np.save('/output/unmask_preds.npy', unmask_preds)
```
