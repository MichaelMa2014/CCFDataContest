# 大数据精准营销中搜狗用户画像挖掘

[大数据精准营销中搜狗用户画像挖掘](http://www.datafountain.cn/data/science/player/competition/detail/description/239)

## 传统模型

- [scikit-learn](http://scikit-learn.org/)
- [XBgoost](http://xgboost.readthedocs.io/en/latest/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Extra Trees|0.540878584503|0.803474706183|0.582671957672|0.642341749453|/
Logistic Regression|0.568232662192|0.808584568217|0.612764550265|0.663193926891|/
Multinomial Naive Bayes|0.540370144397|0.794992335207|0.569885361552|0.635082613719|/
Random Forest|0.543929225137|0.799693408278|0.581900352734|0.641840995383|/
Support Vector Machine|0.566503965833|0.80756259581|0.610339506173|0.661468689272|/
XGBoost|0.537014439699|0.769136433316|0.569223985891|0.625124952969|/

## 神经网络

- [Keras](https://keras.io/)（后端采用[Theano](http://www.deeplearning.net/software/theano/)）【[中文文档](http://keras-cn.readthedocs.io/en/latest/)】
- [scikit-learn](http://scikit-learn.org/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Convolution Neural Networks|/|/|/|/|/
Fast Text (ngram=1) (old)|0.58429936954|0.821461420517|0.616622574956|0.674127788338|0.68878
Multi-Layer Perceptron|0.576774455981|0.814103219189|0.617504409171|0.66946069478|/
Multi-Layer Perceptron (scikit-learn)|0.57240187106|0.815738375064|0.61364638448|0.667262210201|/

## 集成学习

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Stacking Ensemble (Blend) with 8 models|/|/|/|/|/
Voting Ensemble (hard) with 8 models|/|/|/|/|/
Voting Ensemble (soft) with 7 models|/|/|/|/|/

PS: higher is better

各算法具体参数请参考代码
