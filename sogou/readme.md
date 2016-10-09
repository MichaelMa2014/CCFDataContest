# 大数据精准营销中搜狗用户画像挖掘

[大数据精准营销中搜狗用户画像挖掘](http://www.datafountain.cn/data/science/player/competition/detail/description/239)

## 传统模型

- [scikit-learn](http://scikit-learn.org/)
- [TextGrocery](http://textgrocery.readthedocs.io/zh/latest/)
- [XBgoost](http://xgboost.readthedocs.io/en/latest/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Logistic Regression|0.561832061069|0.779877425945|0.583011583012|0.641573690008|0.6485
Multinomial Naive Bayes|0.53893129771|0.779877425945|0.564258135687|0.627688953114|/
Random Forest|0.517048346056|0.787538304392|0.553778268064|0.619454972837|/
Support Vector Machine|0.514503816794|0.745658835546|0.544953116382|0.601705256241|/
TextGrocery|0.542493638677|0.788049029622|0.560948703806|0.630497124035|/
XGBoost|0.519083969466|0.758937691522|0.551571980143|0.609864547044|/

## 神经网络

- [Keras](https://keras.io/)（后端采用[Theano](http://www.deeplearning.net/software/theano/)）【[中文文档](http://keras-cn.readthedocs.io/en/latest/)】

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Convolution Neural Networks|0.537404580198|0.80132788505|0.575841147533|0.63819120426|/
Fast Text (ngram=1)|0.561323155853|0.795709907887|0.59238830697|0.64980712357|0.6638
Multi-Layer Perceptron|0.56234097047|0.791113380149|0.578599011543|0.644017787387|0.6506

## 集成学习

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Stacking Ensemble (Blend) with 4 models (Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost)|0.567938931298|0.789581205312|0.603971318257|0.653830484955|0.6631

PS: higher is better

各算法具体参数请参考代码
