
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import scipy


# In[2]:

def extract_test_and_train_data(output_train_file, output_test_file):
    
    # Read the filtered file 
    df = pd.read_csv('raw/test_rating_filtered.csv')
    
    # Get a list of all the user ids
    users = list(df['user_id'].unique())
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    count = 0
    for u in users:
        df_u = df[df['user_id'] == u]
        train_u = df_u.sample(frac=0.8, random_state=200)
        test_u = df_u.drop(train_u.index)
        df_train = df_train.append(train_u, ignore_index=True)
        df_test = df_test.append(test_u, ignore_index=True)

        count += 1
        if count % 100 == 0:
            print count
    
    df_train.to_csv(output_train_file, index=False, encoding='UTF-8')
    df_test.to_csv(output_test_file, index=False, encoding='UTF-8')


# In[6]:

def filter_raiting_and_winsorizing(output_filtered):
    # Read the original data file
    f_rating = 'raw/rating.csv'
    df_rating = pd.read_csv(f_rating)

    # Remove the movies watched where the rating is below or equal to 5
    rating_5 = df_rating[df_rating['rating'].isin([-1, 6, 7, 8, 9, 10])]

    # Perform 98% winsorizing
    lower_threshold = 0.01
    upper_threshold = 0.99

    # Get the number of movies watched by each user
    users_no_movies = rating_5.groupby('user_id').size().to_frame().sort_values(by=0)

    lower_no_movies = users_no_movies.quantile(lower_threshold).astype(int).values[0]
    higher_no_movies = users_no_movies.quantile(upper_threshold).astype(int).values[0]

    # Reindex the dataframe so they have two columns only: user_id and count of movies
    users_no_movies.reset_index(inplace=True)

    # If the lower number of movies is below 10, then the cut is performed at 10
    if lower_no_movies < 10:
        lower_no_movies = 10

    # Get the ids of the users that should be kept
    users_ids = users_no_movies['user_id'].loc[users_no_movies[0].isin(range(lower_no_movies,higher_no_movies))].astype(int).tolist()

    # Filter the users
    user_ratings = rating_5[rating_5['user_id'].isin(users_ids)]
    
    # Create the filtered data file
    user_ratings.to_csv(output_filtered, encoding='UTF-8', index=False)


# In[7]:

print("Filtering started...")
# Filter the original file according to the preprocessing considerations
filter_raiting_and_winsorizing(output_filtered = 'raw/test_rating_filtered.csv')

print("Filtering done!")

print("Train/Test split started...")

# Separate the filtered file into train and test datasets
extract_test_and_train_data(output_train_file = "raw/rating_train.csv", 
                            output_test_file = "raw/rating_test.csv")

print("Train/Test split done")


# In[ ]:



