import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f_anime = "anime.csv"
f_rating = 'rating.csv'
f_filtered = 'rating_filtered.csv'


def main():
    if len(sys.argv) > 0 and 'filter' in sys.argv:
        filter_raiting('../raw/rating.csv')

    if len(sys.argv) > 0 and 'train' in sys.argv:
        df_rating = pd.read_csv('rating_filtered.csv')
        extract_test_and_train_data(df_rating)

# Filter >5; == -1


def filter_raiting(f_rating):
    df_rating = pd.read_csv(f_rating)
    rating_5 = df_rating[df_rating['rating'].isin([-1, 6, 7, 8, 9, 10])]
    rating_5.to_csv('rating_filtered.csv', encoding='UTF-8', index=False)
#
# Extract test and train data; save them in separate files


def extract_test_and_train_data(df):
    users = list(df['user_id'].unique())
    # First 5 users
    users_5 = users

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    count = 0
    for u in users_5:
        df_u = df[df['user_id'] == u]
        train_u = df_u.sample(frac=0.8, random_state=200)
        test_u = df_u.drop(train_u.index)
        df_train = df_train.append(train_u, ignore_index=True)
        df_test = df_test.append(test_u, ignore_index=True)

        count += 1
        if count % 100 == 0:
            print count

    df_train.to_csv("rating_train.csv", index=False, encoding='UTF-8')
    df_test.to_csv("rating_test.csv", index=False, encoding='UTF-8')
#


if __name__ == "__main__":
    main()
