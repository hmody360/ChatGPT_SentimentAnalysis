#------------------------Initialization0------------------------------------
#importing required data processing modules

import pandas as pd
import numpy as np

# Creating a data frame using the CSV file containing all the gathered tweets using its path
df = pd.read_csv("C:/Users/Owner/Desktop/MachineLeaningFile/file.csv")

# printing the first few lines of the CSV file(Making sure it can read it & to check the input)
df.head()

# -----------------------Data PreProcessing---------------------------------
# Removing the first unamed column(it could mess with the results if read)
df = df[['tweets', 'labels']]

df.head()

#Reading the first 10 rows in the tweets column
for i in df.tweets.head(10):
    print(i)
    print()

#Since these tweets contain links and all the links start with https 
#we'll remove them from the tweets since they can interfere with the results 
#and we only need the tweet's text
df['tweet_list'] = df['tweets'].str.split('https:')

df.head()