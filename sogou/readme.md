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
- [scikit-learn](http://scikit-learn.org/)

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Convolution Neural Networks|0.584732825064|0.821756895278|0.590733590931|0.665741103758|/
Fast Text (ngram=1)|0.581170483582|0.822778345189|0.598455598653|0.667468142475|/
Multi-Layer Perceptron|0.5745547075|0.817160367661|0.590733590997|0.660816222053|/
Multi-Layer Perceptron (scikit-learn)|0.573027989822|0.812563840654|0.585217870932|0.656936567136|/

## 集成学习

algorithm|validation age score|validation gender score|validation education score|validation final score|final score
:-:|:-:|:-:|:-:|:-:|:-:
Stacking Ensemble (Blend) with 8 models|0.596437659033|0.819713993871|0.595697738555|0.67061646382|/
Voting Ensemble (hard) with 8 models|0.565394402036|0.816138917263|0.584666298952|0.65539987275|/
Voting Ensemble (soft) with 7 models|0.572010178117|0.817160367722|0.581908439051|0.657026328297|/

PS: higher is better

各算法具体参数请参考代码
