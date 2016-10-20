# 大数据精准营销中搜狗用户画像挖掘

[大数据精准营销中搜狗用户画像挖掘](http://www.datafountain.cn/data/science/player/competition/detail/description/239)

## 传统模型

- [scikit-learn](http://scikit-learn.org/)
- [TextGrocery](http://textgrocery.readthedocs.io/zh/latest/)
- [XBgoost](http://xgboost.readthedocs.io/en/latest/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Bernoulli Naive Bayes|0.541475826972|0.819203268641|0.507446221732|0.622708439115|/
Extra Trees|0.533842239186|0.814606741573|0.569222283508|0.639223754756|/
Logistic Regression|0.562849872774|0.801838610827|0.57584114727|0.64684321029|/
Multinomial Naive Bayes|0.543511450382|0.805924412666|0.562603419746|0.637346427598|/
Random Forest|0.526208651399|0.79315628192|0.568119139548|0.629161357623|/
Support Vector Machine|0.558269720102|0.804392236977|0.57749586321|0.646719273429|/
TextGrocery|0.541984732824|0.786516853933|0.564258135687|0.630919907481|/
XGBoost|0.532315521628|0.768641470889|0.555984555985|0.618980516167|/

## 神经网络

- [Keras](https://keras.io/)（后端采用[Theano](http://www.deeplearning.net/software/theano/)）【[中文文档](http://keras-cn.readthedocs.io/en/latest/)】

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Fast Text (ngram=1)|0.58167938998|0.822778345737|0.597904026673|0.667453920797|/
Multi-Layer Perceptron|0.574045797705|0.813585295193|0.587424156551|0.658351749816|/

## 集成学习

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Stacking Ensemble (Blend) using ExtraTrees with 7 models (Bernoulli Naive Bayes, ExtraTrees, Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost, Multi-Layer Perceptron)|0.590330788804|0.819203268641|0.585217870932|0.664917309459|/
Stacking Ensemble (Blend) using Logistic Regression with 7 models (Bernoulli Naive Bayes, ExtraTrees, Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost, Multi-Layer Perceptron)|0.598473282443|0.819203268641|0.601213458356|0.67296333648|0.6725
Voting Ensemble (hard) with 7 models (Bernoulli Naive Bayes, ExtraTrees, Logistic Regression, Multinomial Naive Bayes, Random Forest, Support Vector Machine, XGBoost)|0.558269720102|0.819713993871|0.585769442912|0.654584385628|/
Voting Ensemble (soft) with 6 models (Bernoulli Naive Bayes, ExtraTrees, Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost)|0.569974554707|0.819203268641|0.583563154992|0.657580326114|/

PS: higher is better

各算法具体参数请参考代码
