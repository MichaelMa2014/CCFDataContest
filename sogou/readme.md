# 大数据精准营销中搜狗用户画像挖掘

[大数据精准营销中搜狗用户画像挖掘](http://www.datafountain.cn/data/science/player/competition/detail/description/239)

### 传统模型

- [scikit-learn](http://scikit-learn.org/)
- [TextGrocery](http://textgrocery.readthedocs.io/zh/latest/)
- [XBgoost](http://xgboost.readthedocs.io/en/latest/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Bernoulli Naive Bayes|0.518913971934|0.797956055187|0.505511463845|0.607460496988|/
Extra Trees|0.536099247509|0.800613183444|0.577380952381|0.638031127778|/
Logistic Regression|0.561928004881|0.804394481349|0.606701940035|0.657674808755|/
Multinomial Naive Bayes|0.539861704291|0.791415431783|0.57032627866|0.633867804911|/
Random Forest|0.53711612772|0.798978027593|0.581569664903|0.639221273405|/
Support Vector Machine|0.557148667887|0.803576903424|0.604607583774|0.655111051695|/
XGBoost|0.531726662599|0.764946346449|0.562830687831|0.619834565626|/

### 神经网络

- [Keras](https://keras.io/)（后端采用[Theano](http://www.deeplearning.net/software/theano/)）【[中文文档](http://keras-cn.readthedocs.io/en/latest/)】
- [scikit-learn](http://scikit-learn.org/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Convolution Neural Networks|/|/|/|/|/
Fast Text (ngram=1)|/|/|/|/|/
Multi-Layer Perceptron|0.575249135664|0.810424118567|0.610559964727|0.665411072986|/
Multi-Layer Perceptron (scikit-learn)|0.570469798658|0.811241696474|0.608245149912|0.663318881681|/

### 集成学习

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Stacking Ensemble (Blend) with 8 models|/|/|/|/|/
Voting Ensemble (hard) with 8 models|/|/|/|/|/
Voting Ensemble (soft) with 7 models|/|/|/|/|/

PS: higher is better

各算法具体参数请参考代码
