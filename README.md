### Project: 数云 Jira Issue 类型分类

### Highlights:
  - 这是一个 **multi-class text classification (sentence classification)** 问题
  - 目标是 **将数云 Jira 中的 Issue 分类到 119 种 Issue Type 中**.
  - 模型为 **CNN, RNN (LSTM and GRU) and Word Embeddings** 用 **Tensorflow** 

### Data: 数云公司 Jira Issue，数据经过了其他程序的预处理，只保留了需要的字段
  - Input: **Issue Description + Issue Summary + Issue Comments**
  - Output: **Issue Type**
  - Examples:
    
    ```xml
    <Issue>
        <id>10000</id>
        <issuetype>Task</issuetype>
        <summary><![CDATA[客人精确查询及跟踪评估的索引优化后开启外网功能]]></summary>
        <description><![CDATA[注意sas2db和hive2db的建索引模板文件也要更新]]></description>
        <comments><![CDATA[所有的精确查询都已修改，也新建了索引。]]></comments>
    </Issue>
    ```

    
### Train:
  - Command: python3 train.py
  - Example: 
    
    ```bash
    python3 train.py
    ```

### Predict:
  - Command: 
  
    ```
    gunicorn -t 3600 predict_web:app
    ```
  - 说明：由于样本不均衡，所以程序中加了随机过采样,但是过采样后的数据集达到上千万，所以出现Memory Error。在此之前精度为80%，目前这段程序结果还没有出来，正在解决中。。。


  
### Reference:
- [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-with-tensorflow/)
- [Github of implement a CNN-RNN for text multi-classification with tensorflow](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)
