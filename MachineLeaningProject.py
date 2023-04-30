#------------------------Initialization0------------------------------------
#importing required Data/String processing modules

import pandas as pd
import numpy as np
import re
#importing data visualization modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
# importing Scikit learn models for classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
# Creating a data frame using the CSV file containing all the gathered tweets using its path
df = pd.read_csv("C:/Users/Owner/Desktop/MachineLeaningFile/file.csv")

# printing the first few lines of the CSV file(Making sure it can read it & to check the input)
df.head()

# -----------------------Data PreProcessing---------------------------------
# Removing the first unamed column(it could mess with the results if read)
df = df[['tweets', 'labels']]

df.head()

# Reading the first 10 rows in the tweets column
for i in df.tweets.head(10):
    print(i)
    print()

# Since these tweets contain links and all the links start with https 
# we'll remove them from the tweets since they can interfere with the results 
# and we only need the tweet's text
df['tweet_list'] = df['tweets'].str.split('https:')

df.head()
# Selecting only the text from tweets and inputting it into a variable

text = [i[0] for i in df.tweet_list]
df['text'] = text
df = df[['text', 'labels']]
df.head()

# Removing all characters that aren't a number or in the alphabet from the text
# setting up regex
string = r'[A-Za-z0-9]'
# method to remove all non-alphaneumric characters
trim_list = []

for row in text:
    s=''
    for letter in row:
        if bool(re.match(string, letter)):
            s+=letter
    trim_list.append(s)

# Removing all non-printing characters from text

rep_list = ['\U0001fae1', '\\n','@','#','\xa0','***']
for i in trim_list:
    for j in rep_list:
        if j in i:
            i.replace(j,'')

df['text'] = trim_list

df.head()
# -----------------------Data Visualization---------------------------------
# #initializing comment and stop words variables
# comment_words = ""
# stopwords = set(STOPWORDS)

# #method to get each word seperately
# for val in df.text:
#     val = str(val)
#     tokens = val.split()

#     for i in range(len(tokens)):
#         tokens[i] = tokens[i].lower()

#     comment_words += " ".join(tokens)+" "

# #setting up mask to visualize data on
# mask = np.array(Image.open("C:/Users/Owner/Desktop/MachineLeaningFile/masks/comment.png"))

# #setting up the wordcloud
# wordcloud = WordCloud(width = 800, height = 800,
#             background_color ='pink', 
#             stopwords= stopwords,
#             min_font_size = 10, mask = mask).generate(comment_words)
# #plotting the WordCloud Image
# plt.figure(figsize=(8,8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)

# plt.show()
# -----------------------Model Selection---------------------------------
#labels to integers guide
# 1 = Good
# 0 = netural
#-1 = bad
df['lab_int'] = np.where(df['labels']=='good',1,np.where(df['labels']=='bad',-1,0))
#creating train and test variables
X_train,X_test,y_train,y_test = train_test_split(df['text'],df['lab_int'],test_size=0.3,random_state=1)
#converting the text document into a matrix of tokens
vec = CountVectorizer(ngram_range=(1,3),stop_words="english")
#fitting the vectorized sets into the variables
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
#Applying Navie Bayas Model
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)

preds = nb.predict(X_test)
print(classification_report(y_test,preds))

# using logistic regression method
log = LogisticRegression()
log.fit(X_train,y_train)

preds = log.predict(X_test)
print(classification_report(y_test,preds))
# -----------------------Hyper Parameter Tuning---------------------------------
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold

param_grid = {"alpha":[0.1,0,1.0,10,100]}

grid_search = GridSearchCV(MultinomialNB(), param_grid,verbose=2)
grid_search.fit(X_train,y_train)

grid_search.best_params_