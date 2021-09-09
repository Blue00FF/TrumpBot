from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pyinputplus as pyip

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

my_bot = ChatBot(name='TrumpBot', read_only=True, logic_adapters=[
                 'chatterbot.logic.BestMatch'])

df = pd.read_csv("./trump_insult_tweets_2014_to_2021.csv")
sentences = "\n".join(df['tweet'].to_list())

sentence_tokens = nltk.sent_tokenize(sentences)


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


stop_words = set(stopwords.words("english"))
tokenizer = LemmaTokenizer()
token_stop = tokenizer(" ".join(stop_words))


def generateResponse(user_response):
    user_response = user_response.lower()
    Trump_response = ""
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    vectors = vectorizer.fit_transform([user_response] + sentence_tokens)
    cosine_similarities = linear_kernel(vectors[0:1], vectors)
    related_indices = cosine_similarities.argsort()[0][-1]
    flat = cosine_similarities.flatten()
    flat.sort()
    req = flat[-1]
    if req == 0:
        print("I don't know what to say to that.")
    else:
        Trump_response += sentence_tokens[related_indices]
    return Trump_response


#print("Type bye when you want to quit the program.")
#print("Why are you here?")
print(generateResponse("What do you think about television?"))

# while True:
#    user_response = pyip.inputStr()
#    if user_response.lower() != "bye":
#        print(generateResponse(user_response))
#        print("TrumpBot : ", end="")
#        print(generateResponse(user_response))
#        sent_tokens.remove(user_response)
#    else:
#        break

#hprint("Bye, you son of a gun!")
