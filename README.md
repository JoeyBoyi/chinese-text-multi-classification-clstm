# Multi-class Text Classification
Implement four neural networks in Tensorflow for multi-class text classification problem.

## Models
* A LSTM classifier. See rnn_classifier.py
* A Bidirectional LSTM classifier. See rnn_classifier.py
* A CNN classifier. See cnn_classifier.py. Reference: [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
* A C-LSTM classifier. See clstm_classifier.py. Reference: [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630).

## Data Format
Training data should be stored in csv file. The first line of the file should be ["label", "content"] or ["content", "label"].

## Requirements

- Python 3.5 or 3.6
- Tensorflow >= 1.4.0
- Numpy

## Train
Run train.py to train the models.
Parameters:
```
python train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --clf CLF             Type of classifiers. Default: cnn. You have four
                        choices: [cnn, lstm, blstm, clstm]
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --language LANGUAGE   Language of the data file. You have two choices: [ch,
                        en]
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Max document length
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Cross validation test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. For CNN, C-LSTM.
  --filter_sizes FILTER_SIZES
                        CNN filter sizes. For CNN, C-LSTM.
  --num_filters NUM_FILTERS
                        Number of filters per filter size. For CNN, C-LSTM.
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell. For LSTM, Bi-
                        LSTM
  --num_layers NUM_LAYERS
                        Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM
  --keep_prob KEEP_PROB
                        Dropout keep probability
  --learning_rate LEARNING_RATE
                        Learning rate
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate the model on validation set after this many
                        steps
  --save_every_steps SAVE_EVERY_STEPS
                        Save the model after this many steps
  --num_checkpoint NUM_CHECKPOINT
                        Number of models to store
```
**You could run train.py to start training**:

```
 python3 train.py --data_file=your_dataFile_path --clf=clstm --language=ch --num_classes=118 --vocab_size=20000

```

**You could run train_test.py to start training and test**:
```
 python3 train_test.py --data_file='/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/data/SearchRequest_All.xml' --clf=clstm --language=ch --num_classes=118 --vocab_size=20000 --num_epochs=30
```


## Pridect

```
python3 pridect.py --file_path=./runs/1516344401/ --model_file=clf-65000
```

## Description
* 数据只选择了summary和description部分做为输入初始样本，经过预处理后，去除掉英文、数字和不必要的符号串，最后样本中只剩下中文内容，作为最终的分词文本集
* 最终使用CLSTM模型对118个标签进行分类得到精度为83.94%.  
*PS: 上次有个小bug，所以导致精度达到86%，经过处理后，上面是目前最终结果；比原来RCNN模型高了4%左右*

## Second update
* 这次更新把原来的data.py删除，数据处理都在data_helper.py文件中
* 增加了train_test.py，将数据集分成训练集、开发集(验证集)、测试集。而原来train.py只有训练集和验证集
* 增加了pridect.py文件，对未登录词进行预测。
***PS：我拿了其他人最近刚提交的Issue，测试了十几个，效果很好***

## Third update
* 本次更新data_help.py文件，主要内容如下：
  
  * 为了解决类别不均衡问题(最多类别十万多，而最少类别只有个位数，具体见分支2 dataOut文件夹下的labelCouter.txt文件)，有几种方法，本项目选择随机过采样方法使得类别达到相对均衡(还有其他的一些方法转[类别不均衡解决办法](http://blog.csdn.net/heyongluoyao8/article/details/49408131))，这可能不是最好的解决办法，但它是work。
  
  * 最后精度达到89.50%，运行方法同上。较之前提高了6%。


## TODOs
* The F1-score or Precision or Recall metrics would be computed with the model as the unbalanced category in the text dataset. 
* The parameters would be tuned by the next time.

## Reference 
* [Text Classification with CLSTM or CNN or RNN](https://github.com/zackhy/TextClassification)