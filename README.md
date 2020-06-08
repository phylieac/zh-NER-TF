# Ⅰ. Chanages
# 1. PlaceHolder Inputs
```python
self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
# self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
self.lr_pl=tf.Variable(initial_value=self.lr,name='lr',trainable=False)
```

# 2. Add CRF Decode Layer
```python
with tf.variable_scope("crf_decode"):
    self.best_score,_=tf.contrib.crf.crf_decode(self.logits,self.transition_params,self.sequence_lengths)
```

# 3. Identify Outputs
```python
tf.identity(self.best_score, name="output_labels")
```

# 4. CRF Transition_params
```python
self.transition_params = tf.Variable(initial_value=self.transition_params,name='transition_params',trainable=False)
```

# 4. Save .pb Format Model
```python
tf.saved_model.simple_save(sess,self.model_path,inputs={"word_ids":self.word_ids,"dropout":self.dropout_pl,"sequence_lengths":self.sequence_lengths},outputs={"best_score":self.best_score})
```

# New NER model 

## which data?
下载地址：https://github.com/CLUEbenchmark/CLUE

本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.

训练集：10748 验证集：1343

标签类别：
数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

cluener下载链接：[数据下载](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip)

## .pb model
data_path_save/1591586134/checkpoints/model/

## how c++ use?
> C++ 调用.pb model 项目，依赖libtensorflow_cc.so动态库与tensorflow include head file.
> [xcode C++ project: TF-NER](https://github.com/phylieac/TF-NER.git)



# Ⅱ. Belown is Original Author's  Readme.txt

> # A simple BiLSTM-CRF model for Chinese Named Entity Recognition

> This repository includes the code for buliding a very simple __character-based BiLSTM-CRF sequence labeling model__ for Chinese Named Entity Recognition task. Its goal is to recognize three types of Named Entity: PERSON, LOCATION and ORGANIZATION.

> This code works on __Python 3 & TensorFlow 1.2__ and the following repository [https://github.com/guillaumegenthial/sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging) gives me much help.

> ## Model

> This model is similar to the models provided by paper [1] and [2]. Its structure looks just like the following illustration:

> ![Network](./pics/pic1.png)

> For one Chinese sentence, each character in this sentence has / will have a tag which belongs to the set {O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}.

> The first layer, __look-up layer__, aims at transforming each character representation from one-hot vector into *character embedding*. In this code I initialize the embedding matrix randomly. We could add some linguistic knowledge later. For example, do tokenization and use pre-trained word-level embedding, then augment character embedding with the corresponding token's word embedding. In addition, we can get the character embedding by combining low-level features (please see paper[2]'s section 4.1 and paper[3]'s section 3.3 for more details).

> The second layer, __BiLSTM layer__, can efficiently use *both past and future* input information and extract features automatically.

> The third layer, __CRF layer__,  labels the tag for each character in one sentence. If we use a Softmax layer for labeling, we might get ungrammatic tag sequences beacuse the Softmax layer labels each position independently. We know that 'I-LOC' cannot follow 'B-PER' but Softmax doesn't know. Compared to Softmax, a CRF layer can use *sentence-level tag information* and model the transition behavior of each two different tags.

> ## Dataset

> |    | #sentence | #PER | #LOC | #ORG |
> | :----: | :---: | :---: | :---: | :---: |
> | train  | 46364 | 17615 | 36517 | 20571 |
> | test   | 4365  | 1973  | 2877  | 1331  |

> > It looks like a portion of [MSRA corpus](http://sighan.cs.uchicago.edu/bakeoff2006/). I downloaded the dataset from the link in `./data_path/original/link.txt`

> ### data files

> The directory `./data_path` contains:

> - the preprocessed data files, `train_data` and `test_data` 
> - a vocabulary file `word2id.pkl` that maps each character to a unique id  

> For generating vocabulary file, please refer to the code in `data.py`. 

> ### data format

> Each data file should be in the following format:

> ```
> 中	B-LOC
> 国	I-LOC
> 很	O
> 大	O
> 
> 句	O
> 子	O
> 结	O
> 束	O
> 是	O
> 空	O
> 行	O
> 
> ```

> If you want to use your own dataset, please: 

> - transform your corpus to the above format
> - generate a new vocabulary file

> ## How to Run

> ### train

`python main.py --mode=train `

> ### test

> `python main.py --mode=test --demo_model=1521112368`

> Please set the parameter `--demo_model` to the model that you want to test. `1521112368` is the model trained by me. 

> An official evaluation tool for computing metrics: [here (click 'Instructions')](http://sighan.cs.uchicago.edu/bakeoff2006/)

> My test performance:

> | P     | R     | F     | F (PER)| F (LOC)| F (ORG)|
> | :---: | :---: | :---: | :---: | :---: | :---: |
> | 0.8945 | 0.8752 | 0.8847 | 0.8688 | 0.9118 | 0.8515

> ### demo

> `python main.py --mode=demo --demo_model=1521112368`

> You can input one Chinese sentence and the model will return the recognition result:

> ![demo_pic](./pics/pic2.png)

> ## Reference

> \[1\] [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991v1.pdf)
> > 
> \[2\] [Neural Architectures for Named Entity Recognition](http://aclweb.org/anthology/N16-1030)
> 
> \[3\] [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](https://> link.springer.com/chapter/10.1007/978-3-319-50496-4_20)

> \[4\] [https://github.com/guillaumegenthial/sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)  
