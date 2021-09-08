from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pandas as pd

my_bot = ChatBot(name='TrumpBot', rad_only=True, logic_adapters=[
                 'chatterbot.logic.BestMatch'])

df = pd.read_csv("./trump_insult_tweets_2014_to_2021.csv")

corpus_trainer = ListTrainer(my_bot)
corpus_trainer.train(df['tweet'].to_list())

print(my_bot.get_response("Hello"))
