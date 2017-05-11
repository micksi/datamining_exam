import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

f_anime = "anime.csv"
f_rating = 'rating.csv'
f_filtered = 'rating_filtered.csv'

# Filter >5; == -1
def filter_raiting(f_rating):
    df_rating = pd.read_csv(f_rating)
    rating_5 = df_rating[df_rating['rating'].isin([-1, 6, 7, 8, 9, 10])]
    rating_5.to_csv('rating_filtered.csv', encoding='UTF-8', index = False)
#
# Extract test and train data; save them in separate files
def extract_test_and_train_data(df):
    users = list(df['user_id'].unique())
    # First 5 users
    users_5 = users[:5]

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for u in users_5:
        df_u = df[df['user_id'] == u]    
        train_u = df_u.sample(frac=0.8, random_state=200)
        test_u = df_u.drop(train_u.index)    
        df_train = df_train.append(train_u, ignore_index = True)
        df_test = df_test.append(test_u, ignore_index = True)

    df_train.to_csv("rating_train.csv", index = False, encoding='UTF-8')
    df_test.to_csv("rating_test.csv", index = False, encoding='UTF-8')
#

df_filtered = pd.read_csv(f_filtered)
extract_test_and_train_data(df_filtered)