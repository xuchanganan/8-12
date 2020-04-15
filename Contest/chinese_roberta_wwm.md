# 环境
python 1.12
# utils.py
```
import random
import numpy as np

class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.t1 = []
        self.t2 = []
        self.ys = []
        self.weight = []
        
def createBatch(samples, max_x_length):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    def tranf_label(y, MAX_LABELS=3):
        label = np.zeros(MAX_LABELS)
        label[int(y)]=1
        return list(label)
    
    batch = Batch()
    for sample in samples:
        #将source PAD值本batch的最大长度
        batch.t1.append( list(sample[0])+[0] * (max_x_length - len(list(sample[0]))) )
        batch.t2.append( list(sample[1])+[0] * (max_x_length - len(list(sample[1]))) )
        batch.ys.append(tranf_label(sample[2]))
        batch.weight.append( sample[3] )
        
    return batch

def getBatches( data, batch_size, max_x_length ):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是Q-A对的列表
    :param batch_size: batch大小
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    # 根据自己情况决定, 是否每个epoch之前都要进行样本的shuffle.
    # random.shuffle(data) 
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples, max_x_length)
        batches.append(batch)
    return batches
```

# model.py
```
from bert import modeling

def load_bert(input_ids, input_mask):
    # 这个位置. 注意修改. 
    bert_config_file = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    
    # 初始化BERT
    model = modeling.BertModel( config=bert_config,
                                is_training=False,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                use_one_hot_embeddings=False
                                )
    bert_output = model.get_sequence_output()
    
    return bert_output



# 构建模型
class Classification(object):
    """
    Text RNN 用于文本分类
    """
    def __init__(self, config):

        self.lr = config.learning_rate
        
        self.batch_size = config.batchSize
        self.max_x_length = config.max_x_length
        self.dropout_prob = 0.8
        
        # 定义模型的输入
        self.t1_ = tf.placeholder(tf.int32, [None, self.max_x_length], name="input_t1")
        self.t2_ = tf.placeholder(tf.int32, [None, self.max_x_length], name="input_t2")
        self.ys = tf.placeholder(tf.float32, [None,3], name="input_c")
        self.weight = tf.placeholder(tf.float32, [None], name="weight")
#        
        # bert model
        t_ = load_bert( self.t1_,self.t2_ )
        
        first_token_tensor = tf.concat([
                tf.squeeze(t_[-4][:, 0:1, :], axis=1),
                tf.squeeze(t_[-3][:, 0:1, :], axis=1),
                tf.squeeze(t_[-2][:, 0:1, :], axis=1),
                tf.squeeze(t_[-1][:, 0:1, :], axis=1)], axis=1)
        bert_pool_t = tf.layers.dense(
            first_token_tensor,
            768,
            activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        
        pool_t = self.dynamic_attention(t_[-1])
        
        pool_output = tf.concat([ bert_pool_t,pool_t ],1)
        outputSize = pool_output.shape[1].value
        
        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.get_variable(
                    'fc_w', shape=[outputSize, 3], 
                    initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable( tf.constant(0.1, shape=[3]), name='fc_b' )
            self.logits = tf.matmul(pool_output, fc_w) + fc_b
            self.predictions = tf.argmax(self.logits, 1, name='predictions')
            
        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            # Keeping track of l2 regularization loss
            self.loss = tf.reduce_mean( 
                    self.weight * tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.logits, labels=self.ys))

        # Create optimizer
        with tf.name_scope('optimization'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            starter_learning_rate = self.lr
            grad_clip = 5.0
            
            learning_rate = tf.train.exponential_decay(
                    starter_learning_rate, self.global_step, 100, 0.85, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            self.train_op = optimizer.apply_gradients(
                    zip(gradients, variables), global_step=self.global_step)
    
        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.ys, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)



    def dynamic_attention(self, inputs, attention_size=300):
        
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        
        inputs_shape = inputs.shape
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer
        sequence_length = inputs_shape[1].value
        inputs_ = tf.reshape(inputs, [-1, hidden_size])
        
        # Attention mechanism
        W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        b_ = tf.reshape(b, [1, -1])
        
        # v是对所有训练样本的 word--list 进行词向量的 全连接层处理【词向量维度降低】
        v = tf.tanh(tf.matmul(inputs_, W) + b_)
             
        '''
        ## vu 是对所有训练样本的 词向量  转化为1维向量【词向量维度降低】
        vu 学习到的是 所有词的attention权重
        '''
        vu = tf.matmul(v, tf.reshape(u, [-1, 1]))#[batch_size*seq_len,1]
        #[batch_size,seq_len]
        vu_ = tf.exp(vu)
        
        # exps中的 -1 代表 batch_size
        exps = tf.reshape(vu_, [-1, sequence_length])#[batch_size,seq_len]
        # exps_：统计每个 训练样本的 权重和
        exps_ = tf.reduce_sum(exps, 1)
        alphas = exps / tf.reshape(exps_, [-1, 1])#[batch_size,seq_len]
        weight = tf.reshape(alphas, [-1, sequence_length, 1])
        
        # 按照 词向量维度  对所有词 进行 求和（1）
        output = tf.reduce_sum(inputs * weight, 1)#[batch_size,hidden_size]
        
        return output
```
# roberta_wwm.py
```
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import getBatches
from bert import tokenization
from bert import modeling
import os
from model import Classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import optimize


def convert_single_example(tokenizer, text_a):
    tokens_a = tokenizer.tokenize(text_a)
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    tokens = ["[CLS]"]
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将中文转换成ids
    input_mask = [1] * len(input_ids)  # 创建mask

    return tokens, input_ids, input_mask  # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数


def process(data, args, tokenizer, w=1):  # 这里的 w1 是数据的权重.
    all_data = []
    for d in tqdm(data.iterrows()):
        d = d[1]
        text = str(d['微博中文内容'])
        text = text[:args.max_x_length - 2]
        tokens, t1, t2 = convert_single_example(tokenizer, text)
        c = int(d['情感倾向'])
        all_data.append([t1, t2, c, w])

    return all_data


def process_test(data, args, tokenizer):
    # 数据预处理
    T1 = []
    T2 = []
    for d in tqdm(data.iterrows()):
        d = d[1]
        text = str(d['微博中文内容'])
        text = text[-(args.max_x_length - 2):]
        tokens, t1, t2 = convert_single_example(tokenizer, text)
        T1.append(list(t1) + [0] * (args.max_x_length - len(list(t1))))
        T2.append(list(t2) + [0] * (args.max_x_length - len(list(t2))))

    return T1, T2


def predict_test(T1, T2, model):
    rs = []
    bs = 200
    num = int(len(T1) / bs) + 1
    for n in tqdm(range(num)):
        T1_ = T1[n * bs:(n + 1) * bs]
        T2_ = T2[n * bs:(n + 1) * bs]
        if len(T1_) > 0:
            feed = {model.t1_: T1_, model.t2_: T2_}
            _c = sess.run([model.logits], feed_dict=feed)[0]
            rs.extend(_c)
    return np.array(rs)


def TransLabel(x):
    if x == '-1':
        return 0
    elif x == '0':
        return 1
    else:
        return 2


class Config(object):
    batchSize = 32
    epoches = 1
    model_dir = '/output/'


def optF1Score(pt, y):
    pt = np.array(pt)

    def fun(x):
        tmp = np.hstack(
            [x[0] * pt[:, 0].reshape(-1, 1),
             x[1] * pt[:, 1].reshape(-1, 1),
             x[2] * pt[:, 2].reshape(-1, 1)])
        return - f1_score(np.array(y), np.argmax(tmp, axis=1), average='macro')

    initW = np.asarray((1, 1, 1))

    return optimize.fmin_powell(fun, initW)


bert_vocab_file = '/data/changan/chinese_roberta_wwm_ext_L-12_H-768_A-12/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=False)    # tokenization 是导自bert的.
Train_corpus = pd.read_excel('/data/changan/train_dataset/nCoV_100k_train.labled.xlsx')
Train_corpus = Train_corpus[Train_corpus['情感倾向'].isin(['-1', '0', '1'])]
Train_corpus['情感倾向'] = Train_corpus['情感倾向'].apply(lambda x: TransLabel(x))
Train_corpus = Train_corpus.reset_index(drop=True)     # 去掉 旧索引.  reset_index 是 pandas 的方法.
max_x_length = 140
args = Config()
args.max_x_length = max_x_length
args.learning_rate = 0.00002
num_epoches = args.epoches

test_reviews = pd.read_excel('/data/changan/test_dataset/nCov_10k_test.xlsx').fillna('')
T1, T2 = process_test(test_reviews, args, tokenizer)
unmask_reviews = pd.read_csv('/data/changan/train_dataset/nCoV_900k_train.unlabled.csv')[:500000].fillna('')
unmask_T1, unmask_T2 = process_test(unmask_reviews, args, tokenizer)

k_folds = 5
rs = pd.DataFrame()

gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99).split(X=Train_corpus['微博中文内容'].fillna('-1'), y=Train_corpus['情感倾向'].fillna('-1'))
tf.reset_default_graph()
for fold, (train_idx, valid_idx) in enumerate(gkf):

    print('\n', fold + 1)
    if fold > 0:
        tf.reset_default_graph()    # 这里写了重置图结构.

    print('\n...training...')
    model = Classification(args)
    # 加载bert模型
    data_root = '/data/changan/chinese_roberta_wwm_ext_L-12_H-768_A-12/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
    init_checkpoint = data_root + 'bert_model.ckpt'
    tvars = tf.trainable_variables()
    (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_data = pd.DataFrame(Train_corpus, index=list(train_idx))
    train_data = process(train_data, args, tokenizer, 1.0)  # 1-2之间

    dev_data = pd.DataFrame(Train_corpus, index=list(valid_idx))
    dev_data = process(dev_data, args, tokenizer, 1)

    #        train_data += unlabled
    #        np.random.shuffle(train_data)
    batchesTrain = getBatches(train_data, args.batchSize, args.max_x_length)
    batchesDev = getBatches(dev_data, args.batchSize, args.max_x_length)
    current_step = 0

    for e in range(num_epoches):
        print("----- Epoch {}/{} -----".format(e + 1, num_epoches))

        loss_ = 0
        acc_ = 0
        ln_ = len(batchesTrain)
        for nextBatch in tqdm(batchesTrain, desc="Training"):
            feed = {
                model.t1_: nextBatch.t1,
                model.t2_: nextBatch.t2,
                model.ys: nextBatch.ys,
                model.weight: nextBatch.weight,
            }
            _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy], feed_dict=feed)
            current_step += 1
            loss_ += loss
            acc_ += acc
        tqdm.write("----- Step %d -- train_loss %.4f-- acc %.4f" % (
            current_step, loss_ / len(batchesTrain), acc_ / len(batchesTrain)))

        model_dir = args.model_dir + 'fake_{}/bert_cv_{}'.format(fold + 1, fold + 1)
        model.saver.save(sess, model_dir, global_step=current_step)

        loss_ = 0
        acc_ = 0
        ln_ = len(batchesDev)
        for nextBatch in tqdm(batchesDev, desc="Deving"):
            feed = {
                model.t1_: nextBatch.t1,
                model.t2_: nextBatch.t2,
                model.ys: nextBatch.ys,
                model.weight: nextBatch.weight,
            }
            loss, acc, p = sess.run([model.loss, model.accuracy, model.predictions], feed_dict=feed)
            current_step += 1
            loss_ += loss
            acc_ += acc

            # 预测每一折的验证集结果
            if e == num_epoches - 1: 
            # 只有在最后一次epoch才会预测验证集. 
                feed = {model.t1_: nextBatch.t1, model.t2_: nextBatch.t2}
                _c = sess.run([model.logits], feed_dict=feed)[0]
                rs_ = pd.DataFrame(_c, columns=['p1', 'p2', 'p3'])
                rs_['y'] = np.argmax(nextBatch.ys, axis=1)
                rs = pd.concat([rs, rs_])

        tqdm.write("----- Step %d -- dev_loss %.4f-- acc %.4f" % (
            current_step, loss_ / len(batchesDev), acc_ / len(batchesDev)))

    # 预测结果
    c_ = predict_test(T1, T2, model)
    u_ = predict_test(unmask_T1, unmask_T2, model)
    if fold == 0:
        rs_c = c_
        rs_u = u_
    else:
        rs_c += c_
        rs_u += u_

rs.to_csv('/output/fake_opt-f1score.csv', index=False, encoding='utf8')
oof_train = pd.read_csv('/output/fake_opt-f1score.csv')
pWeight = optF1Score(oof_train[['p1', 'p2', 'p3']], oof_train['y'])

f1_valid_preds = oof_train.values * np.concatenate((pWeight, np.array([1])), axis=0)
np.save('/output/fake_f1_valid_preds.npy', f1_valid_preds)


rs_c = rs_c / k_folds * pWeight
rs_u = rs_u / k_folds * pWeight
np.save('/output/fake_f1_average_test_preds.npy', rs_c)
np.save('/output/fake_f1_average_unmask_preds.npy', rs_u)
cc_ = np.argmax(rs_c, 1)

submit = test_reviews[['微博id']]
submit['id'] = submit['微博id']
del submit['微博id']
submit['y'] = [i - 1 for i in cc_]
submit[['id', 'y']].to_csv('/output/fake_submit_test.csv', encoding='utf-8', index=None)
```
