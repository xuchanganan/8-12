# 冷冻bert, 上接BiGRU+Capsule
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


import os
# 内存大小 动态增长
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"  


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


def squash(x, axis=-1):
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(x), axis, keepdims=True)
    scale = tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return x / scale


# Capsule模型构建, 代替池化层
class Capsule(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = tf.keras.backend.conv1d(u_vecs, kernel=self.W)
        else:
            u_hat_vecs = tf.keras.backend.local_conv1d(u_vecs, kernel=self.W, kernel_size=[1], strides=[1])

        batch_size = tf.shape(u_vecs)[0]
        input_num_capsule = tf.shape(u_vecs)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, [batch_size, input_num_capsule, self.num_capsule, self.dim_capsule])
        u_hat_vecs = tf.transpose(u_hat_vecs, perm=[0, 2, 1,
                                                    3])
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = tf.transpose(b, perm=[0, 2, 1])  # shape = [None, input_num_capsule, num_capsule]
            c = tf.nn.softmax(b)  # shape = [None, input_num_capsule, num_capsule]
            c = tf.transpose(c, perm=[0, 2, 1])  # shape = [None, num_capsule, input_num_capsule]
            s_j = tf.reduce_sum(tf.multiply(tf.expand_dims(c, axis=3), u_hat_vecs), axis=2)
            outputs = self.activation(s_j)  # [None,num_capsule,dim_capsule]
            if i < self.routings - 1:
                b = tf.reduce_sum(tf.multiply(tf.expand_dims(outputs, axis=2), u_hat_vecs), axis=3)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def Bert_Gru_Capsule_Model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig.from_pretrained(BERT_PATH + 'bert-base-chinese-config.json', output_hidden_states=True)
    bert_model = TFBertModel.from_pretrained(BERT_PATH + 'bert-base-chinese-tf_model.h5', config=config, name='bert')
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    sequence_output, pooler_output, hidden_states = bert_model(input_id, attention_mask=input_mask,
                                                               token_type_ids=input_atn)

    embedding = hidden_states[-2]
    embed = tf.keras.layers.SpatialDropout1D(0.2)(embedding)
    # rnn 可以尝试 100 128 200 256
    # tf.keras.layers.Bidirectional(
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, activation='sigmoid', return_sequences=True))(  # 这里的激活函数如果是relu, 会出梯度不更新的问题. 
        embed)  # (batch, 1800, 400) 双向是拼接
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)  #
    # (batch,10,16)
    capsule = tf.keras.layers.Flatten()(capsule)  # 拉成(batch, 160)
    x = tf.keras.layers.Dense(1000)(capsule)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(500)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)
    output = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=output)
    return model


def Focal_Loss(y_true, y_pred, alpha=0.5, gamma=2):
    y_pred += tf.keras.backend.epsilon()
    ce = -y_true * tf.math.log(y_pred)
    weight = tf.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = tf.keras.backend.max(fl, axis=-1)
    return reduce_fl


TRAIN_PATH = '/data/changan/train_dataset/'
TEST_PATH = '/data/changan/test_dataset/'
BERT_PATH = '/data/changan/bert_base_chinese/'
MAX_SEQUENCE_LENGTH = 140
input_categories = '微博中文内容'
output_categories = '情感倾向'



df_train = pd.read_csv(TRAIN_PATH+'nCoV_100k_train.labled.csv')
df_train = df_train[df_train[output_categories].isin(['-1','0','1'])]
df_train[input_categories] = df_train[input_categories].fillna(value="无") 

df_test = pd.read_csv(TEST_PATH+'nCov_10k_test.csv')
df_test[input_categories] = df_test[input_categories].fillna(value="无")

df_unmask = pd.read_csv(TRAIN_PATH + 'nCoV_900k_train.unlabled.csv')[:500000]
df_unmask[input_categories] = df_unmask[input_categories].fillna(value="无")


tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-chinese-vocab.txt')
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
unmask_inputs = compute_input_arrays(df_unmask, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = compute_output_arrays(df_train, output_categories)


# Gru_Capsule_Model
Num_capsule = 10
Dim_capsule = 16
Routings = 3

gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))

all_train_idx = [] 
all_valid_idx = [] 

valid_preds = []
test_preds = []
valid_labels = []          # 这里也加了.

unmask_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    all_train_idx.append(train_idx)
    all_valid_idx.append(valid_idx)
    
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = to_categorical(outputs[train_idx])

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = to_categorical(outputs[valid_idx])

    model = Bert_Gru_Capsule_Model()
    
    for layer in model.layers:
        if layer.name == "bert":
            layer.trainable = False
            print(layer.trainable)
    model.summary()  # 查看下是否冷冻了.
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs[0],train_inputs[1],train_inputs[2]),train_outputs)).shuffle(buffer_size=1000).batch(128)
    valid_dataset = tf.data.Dataset.from_tensor_slices(((valid_inputs[0],valid_inputs[1],valid_inputs[2]),valid_outputs)).batch(128)
    
    FL=lambda y_true,y_pred: Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2)
    
    model.compile(loss=FL, optimizer=optimizer, metrics=['acc'])    
    model.fit(train_inputs, train_outputs, validation_data= [valid_inputs, valid_outputs], epochs=2, batch_size=128)
    
    valid_preds.append(model.predict(valid_inputs))
    valid_labels.append(outputs[valid_idx])                    # 这里加了. 
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
