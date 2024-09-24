import numpy as np
import pandas as pd
import seaborn as  sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
#import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from bs4 import BeautifulSoup

data = pd.read_csv('IMDB Dataset.csv')
tokenizer = ToktokTokenizer()
stopwords = nltk.corpus.stopwords.words('english')

def noiseremoval_text(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub(r'\[[^]]*]', '', text)
    return text

data['review'] = data['review'].apply(noiseremoval_text)
#print(data.head()) #1.11.47

def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

data['review'] = data['review'].apply(stemmer)


#removing the stopwords

def removing_stopwords(text, is_lower_case=False):
    tokenizers=ToktokTokenizer();
    tokens = tokenizers.tokenize(text)
    tokens= [i.strip() for i in tokens]
    if is_lower_case:
        filtokens = [i for i in tokens if token not in stopwords]
    else:
        filtokens= [i for i in tokens if i.lower() not in stopwords ]
    filtered_text = ' '.join(filtokens)
    return filtered_text

data['review'] = data['review'].apply(removing_stopwords)
#print(data.head())

#spliting data set from training and testing

train_reviews_data= data.review[:30000]
test_reviews_data = data.review[30000:]

#bag of words

cv= CountVectorizer(min_df=1, max_df=1,binary=False,ngram_range=(1,3))

cv_train = cv.fit_transform(train_reviews_data)
cv_test = cv.transform(test_reviews_data)

print('BOW_cv_train',cv_train.shape)
print('BOW_cv_test',cv_test.shape)

#tf_idf

tf=TfidfVectorizer(min_df=1, max_df=1,use_idf=True,ngram_range=(1,3))
tf_train = tf.fit_transform(train_reviews_data)
tf_test = tf.transform(test_reviews_data)
print('tf_idf_train',tf_train.shape)
print('tf_idf_test',tf_test.shape)

#labeling sentiment data

label = LabelBinarizer()
sentiment_data=label.fit_transform(data['sentiment'])
print(sentiment_data.shape)

train_data=data.sentiment[:30000]
test_data=data.sentiment[30000:]

#training the model
logistic = LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

lr_bow=logistic.fit(cv_train,train_data)
print(lr_bow)

#predicting model for bag of words
bow_predict= logistic.predict(cv_test)
print(bow_predict)

#accuracy score for bag of words
lr_bow_score=accuracy_score(test_data,bow_predict)
print("lr_bow_score : ",lr_bow_score)

#fitting model for tfidf features
lr_tfidf=logistic.fit(tf_train,train_data)
print(lr_tfidf)
lr_tfidf_predict= logistic.predict(tf_test)
print(lr_tfidf_predict)

#accuracy score for tfidf

lr_tfidf_score=accuracy_score(test_data,lr_tfidf_predict)
print("lr_tfidf_score : ",lr_tfidf_score)
