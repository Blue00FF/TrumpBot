import pandas as pd
import nltk
import string
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pyinputplus as pyip


df = pd.read_csv("trump_insult_tweets_2014_to_2021.csv")


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


stop_words = set(stopwords.words("english"))
tokenizer = LemmaTokenizer()

insults = df["insult"].to_list()
tweets = df["tweet"].to_list()
token_stop = tokenizer(" ".join(stop_words))


def generateInsult(user_response):
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    vectors = vectorizer.fit_transform([user_response] + tweets)
    cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
    response_scores = [item.item() for item in cosine_similarities[1:]]
    score_insults = [(score, insult) for score, insult in zip(response_scores, insults)]
    sorted_score_insults = sorted(score_insults, reverse=True, key=lambda x: x[0])
    Trump_response = "You are a " + sorted_score_insults[0][1] + "!"
    return Trump_response


while True:
    user_response = pyip.inputStr(prompt="Tell me about yourself and I will insult you.")
    if user_response != "quit":
        print(generateInsult(user_response))
    else:
        break

print("Goodbye, you son of a gun!")
