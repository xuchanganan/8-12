# 取了bert隐层的后四个CLS
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


def Focal_Loss(y_true, y_pred, alpha=0.5, gamma=2):
    y_pred += tf.keras.backend.epsilon()
    ce = -y_true * tf.math.log(y_pred)
    weight = tf.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = tf.keras.backend.max(fl, axis=-1)
    return reduce_fl


TRAIN_PATH = '/data/changan/train_dataset/'
TEST_PATH = '/data/changan/test_dataset/'
BERT_PATH = '/data/changan//bert_base_chinese/'
MAX_SEQUENCE_LENGTH = 140
input_categories = '微博中文内容'
output_categories = '情感倾向'

df_train = pd.read_csv(TRAIN_PATH + 'nCoV_100k_train.labled.csv')
df_train = df_train[df_train[output_categories].isin(['-1', '0', '1'])]
df_train[input_categories] = df_train[input_categories].fillna(value="无")
df_test = pd.read_csv(TEST_PATH + 'nCov_10k_test.csv')
df_test[input_categories] = df_test[input_categories].fillna(value="无")
df_unmask = pd.read_csv(TRAIN_PATH + 'nCoV_900k_train.unlabled.csv')[:500000]
df_unmask[input_categories] = df_unmask[input_categories].fillna(value="无")

tokenizer = BertTokenizer.from_pretrained(BERT_PATH + 'bert-base-chinese-vocab.txt')
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
unmask_inputs = compute_input_arrays(df_unmask, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = compute_output_arrays(df_train, output_categories)

gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))
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
    train_outputs = to_categorical(outputs[train_idx])
    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = to_categorical(outputs[valid_idx])
    model = WeiboBERT()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_inputs[0], train_inputs[1], train_inputs[2]), train_outputs)).shuffle(buffer_size=1000).batch(32)
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        ((valid_inputs[0], valid_inputs[1], valid_inputs[2]), valid_outputs)).batch(32)
    FL = lambda y_true, y_pred: Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2)
    model.compile(loss=FL, optimizer=optimizer, metrics=['acc'])
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
