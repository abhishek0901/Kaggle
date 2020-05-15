
#************************* TWEETS PREPROCESSING *****************************************
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #Data showing
import string
import nltk

#Read train and test data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('-'*40)
print(f'Some data from train set')
print(train.head())

#tweet column needs to be cleaned

#1. Removing twitter handles
combi = train.append(test,ignore_index=True)

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

#numpy vectorize function take input a function and give its vector form which is numpy aware
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'],"@[\w]*")

#2. Removing punctuations, numbers and special char
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")


#3. removing short words
#Apply method from pandas row by row processing

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

#4&5. Tokenization & Stemming we can do it with spacy as well

tokenized_tweet = combi['tidy_tweet'].apply(lambda x:x.split())
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

#************************* TWEETS PREPROCESSING *****************************************

#************************* STORY GEBERATION AND VISUALISATION ***************************

#1. Wordcloud to understand most frequent words
#Converting all columns into single string

all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud

word_cloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.show()

normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
word_cloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.show()

racist_sexist_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
word_cloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(racist_sexist_words)
plt.figure(figsize=(10,7))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.show()


def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hashtags.append(ht)

    return hashtags

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

#Question -> How this is working
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

#Non racist/sexist Comments
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Racist/Sexist Comments
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
e = e.nlargest(columns="Count", n = 10)
e = e.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#************************* STORY GEBERATION AND VISUALISATION ***************************
