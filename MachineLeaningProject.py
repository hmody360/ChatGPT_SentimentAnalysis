#------------------------Initialization0------------------------------------
#importing required Data/String processing modules

import pandas as pd
import numpy as np
import re
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
df.head

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

df.head