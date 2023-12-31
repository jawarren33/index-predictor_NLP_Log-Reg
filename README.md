# index-predictor_NLP_Log-Reg
This repo constructs a comparative analysis of two popular sentiment analysis tools: Valence Aware Dictionary and Sentiment Reasoner (VADER) and Bidirectional Encoder Representations from Transformers (BERT). Both models use different approaches to learning the data and quantifying sentiment. The VADER model is pretrained on a lexicon dictionary on which it is trained learn the sentiment of the words in the lexicon and translate that into the data fed into it and does not require training. The BERT model on the other hand requires training as it learns to quantify the sentiment based on supervised learning labels. Both approaches are considered as supervised learning approaches as they both require labeled data. Once the sentiment models are compared, the trends in the quantified sentiment are compared to the actual stock index price trends over the same period. The ability for the sentiment models to fit the actual index price trend is assessed using Mean Squared Error (MSE). A Naïve Bayes classifier model is applied to the quantified sentiment trends and is used to classify the following period’s price movement. The accuracy of the Naïve Bayes classifier is assessed based on actual index price movements in the period.

The main objective for this project is to test if sentiment analysis can be applied to quantify intraday stock market sentiment from investor forums to accurately predict market index price movement for the following period. The quantified sentiment is fed into a Naïve Bayes classifier model to classify whether the market index movement in the following period will fall into one of three categories: positive, negative, or neutral. Predicting precise quantity of market index movement would require a number of variables that would not be included in this model that is primarily focused on the impact of investor sentiment on market index movement. Additionally, attempting to predict precise market price changes would cloud the results of the impact of investor sentiment on index movement as it is extremely difficult to accurately predict price movements. A model that can accurately classify the following period market index movement is an invaluable tool for hedge funds and other investors. Previous research has been conducted to predict individual company stock price movements using various sources of sentiment analysis. However, limited research has been performed using StockTwits data to classify near-term stock index movement.

The hypothesis for this project is that there is a linear relationship between the quantified sentiment derived from StockTwits investor forum and the following period stock index price movement. Therefore, the alternate hypothesis is that we can accurately predict intraday stock market movements based on sentiment analysis during the period between market closing and opening with strong accuracy. The null hypothesis is that there is not a linear relationship between the quantified investor sentiment and the actual stock index price movement.
The hypothesized linear relationship between investor sentiment and actual stock index price takes the form of the following where x represents the quantified sentiment and 𝑌 represents the index
price of the next period:

𝑌 =𝛼+𝛽𝑥+𝑒

In conjunction with the hypothesis that there is a quantifiable linear relationship between investor sentiment and stock index price movement, an additional hypothesis is that investor sentiment can be used as an accurate predictor for classifying the movement of the stock index price for the following period in one of three classes: positive, negative, and neutral.

<img width="423" alt="Screen Shot 2023-07-07 at 8 24 01 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/8ffa27bb-a813-4abc-913d-0a6c39e247de">

The following metrics resemble the model’s performance when evaluated on the test dataset:

<img width="628" alt="Screen Shot 2023-07-07 at 8 25 11 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/e1db21b4-aa6a-4622-82c4-38a1d8382a5a">

As we can see, the model performed best when predicting a Bearish tag when the Bearish tag (indicated as a “0”) than the Bullish tag (indicated as a “1”).

<img width="414" alt="Screen Shot 2023-07-07 at 8 26 02 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/c71557a7-3848-4c83-9f66-e8a3918a4243">

The mean value for each day for both pre-market opening, and post-market closing is calculated. In the figure below, the trend of the quantified sentiment as predicted by the BERT model is shown.

<img width="877" alt="Screen Shot 2023-07-07 at 8 27 08 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/8ef550ea-54c1-420e-bde0-bfa3d73ce8ac">

In comparison, the quantified trend predicted by the VADER model is shown below:

<img width="915" alt="Screen Shot 2023-07-07 at 8 30 25 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/f7e6c042-0d13-45f3-96a8-05983a77b58e">

At first glance, the VADER model was slightly more accurate with text classification than the BERT model. Both models, however, we able to generate a similar sentiment trend. The ability of the models to generate a similar sentiment trend is valuable to confirm that the text data being used has the capability to be interpreted as a predictor due to consistency of the trends. In Figure 5 below, the quantified sentiment that has been calculated based on classifications made by the models follows similar trends. To analyze the data on the same scale, both the VADER mean compound score for each period and the BERT mean daily sentiment were standardized using Min-Max scaling. Therefore, the data is standardized on an even scale between 0 and 1.

<img width="900" alt="Screen Shot 2023-07-07 at 8 31 21 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/07a979ff-089c-4deb-80d5-286b09ffdf02">

Initially, one apparent issue with the VADER tool is that it will assign a 100 percent neutral score on phrases that are not of substantial width or contain mostly proper nouns such as names of people or companies. Adjusting for the significant number of neutral statements, all of the statements that have a compound score equal to zero were excluded. Additionally, text that produces a compound score close to zero can be considered indifferentiable. Therefore, only compound scores greater than 0.2 and less that -0.2 were included for analysis. After filtering out the rows of data with indistinguishable compound scores, the sample data then consisted of 8,425 rows of text data. The new distribution of the predicted compound scores is shown below.

<img width="532" alt="Screen Shot 2023-07-07 at 8 28 36 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/35921867-f56c-40af-9ab9-7ce2076cc9f2">


Just by eye, we can see that the quantified sentiment follows almost identical trends for each day. Now we are interested to determine if similar trends arise for each stock index. Below, the three major stock market index price trends over the same period are displayed.
<img width="864" alt="Screen Shot 2023-07-07 at 8 32 17 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/998cb5df-0c00-429a-8742-209a3a26e899">

The stock index data is standardized using Min-Max scaling so that the values can be analyzed in an identical range to the models. The MSE is then calculated for each model on each individual index, results shown below:

<img width="450" alt="Screen Shot 2023-07-07 at 8 33 09 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/ad7cf06f-e23e-4886-a11d-f6cc28c78c6e">

Regardless of having less testing accuracy than the VADER model from a label classification perspective, the BERT model performed better when quantifying sentiment trends that align better with real market price trends. Specifically, the BERT model performed best when aligned with NASDAQ trends. The figure below shows the relationship between the BERT sentiment trend and the NASDAQ price trend.

<img width="873" alt="Screen Shot 2023-07-07 at 8 34 08 PM" src="https://github.com/jawarren33/index-predictor_NLP_Log-Reg/assets/73670838/bfcda570-9f76-481e-837e-304f3904a712">

There lies further room for research. Due to computational and time limitations, the amount of data used to train the BERT and Naïve Bayes models was of small sample size in comparison to industry norms. The models showed improving performance when training with additional epochs. The accuracy of the BERT model improved with both accuracy and a smaller loss curve with every epoch. The Naïve Bayes classified was trained using a sample size that was less than 30 samples, below the industry minimum to statistically reject the null hypothesis that it is not an accurate predictor of stock index movement. However, this research confirmed the foundation for expanding the models used on larger sample sizes and more training epochs. Additionally, only one classifier technique was explored in this research. Other classifier functions could potentially produce better results when trained on the data. (Troussas, et al. 2013) explored the comparative performances of different classifiers and found the Rocchio Classifier to be a classifier that produced a greater recall and F-score than the Naïve Bayes, with a slightly lower accuracy. Further research is required to assess the Rocchio Classifier’s ability to fit to the data specific to this research.
