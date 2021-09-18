# stock-prediction-nlp

This is my Master Thesis working repository.
- I am researching on the capabilities that NLP models have on predicting stock market returns.
- I have chosen Tesla and Bitcoin (not a stock, but does the job), since they are highly volatile and prone to change due to sudden hypes
- I use BERT, a general purpose NLP network developed by google, to predict the sentiment of news articles related to these stocks and trying to find correlations with their returns

Next steps are using conditional Generative Adversarial Networks (cGANs) to combine BERT's sentiment analysis with quantitative techniques to give more accurate predictions. 

> Here you can see the model price predictions for Bitcoin and Tesla based solely on newspaper text

![Bitcoin](https://github.com/juanluisrto/stock-prediction-nlp/blob/master/pngs/predictions_bitcoin.png)

![Bitcoin](https://github.com/juanluisrto/stock-prediction-nlp/blob/master/pngs/predictions_tesla.png)
