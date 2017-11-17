# FDU-NLP-Stock-Market-Prediction
仅描述代码框架，其余理论参见project report。



- Dict：（里面包含了我写project提取特征时需要用到的词典
  - 新闻词汇词典及自己写的将xls转换成我需要的txt格式的代码
    - 帮助jieba分词更准确更专业
  - 频率词.pkl
    - 是我处理的最常出现的词
  - 情绪词
    - 包含了从知网以及大连理工大学实验室下载的情感词汇及程度词否定词等
  - 停用词
    - 帮助预处理与特征降维



- Models：
  - 包括我存好的我训练好的模型（并非最优的模型
- Prediction.py
  - 主要的代码文件
  - 包含了许多函数 由于还不会用class 所以写得比较杂乱 但是每个函数都有注释
  - 主要分成了 预处理 提取特征 训练模型 交叉验证 比较各种模型 挑选最好的模型测试
- ReportTestFile：

  - 包括写report的内容
  - 一些中间输出结果 recall accuracy f1 precision等



