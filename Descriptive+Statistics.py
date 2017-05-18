
# coding: utf-8

# In[345]:

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import scipy
import csv

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

matplotlib.style.use('ggplot')
pd.options.display.float_format = '{:20,.2f}'.format
pd.set_option('display.max_columns', 50)


# In[44]:

data_rating = pd.io.parsers.read_csv('raw/rating.csv')
#data_rating = data_rating.loc[data_rating['user_id'] != 48766]
data_anime = pd.io.parsers.read_csv('raw/anime.csv')

train_rating = pd.io.parsers.read_csv('omer/rating_train.csv')
test_rating = pd.io.parsers.read_csv('omer/rating_test.csv')
final_profiles = pd.io.parsers.read_csv('raw/user_profiles_final.csv')


# In[340]:


def normalize(df_user_profiles):
    x = df_user_profiles.iloc[:, 1:].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)

    df_scaled = pd.DataFrame(x_scaled, columns=df_user_profiles.columns.difference(
        ['user_id', 'rating', 'genre']))

    df_scaled['user_id'] = df_user_profiles['user_id'].values
    df_scaled['genre_count'] = map(
        lambda x: x / 10.0, df_user_profiles['genre_count'].values)
    #df_scaled['rating'] = 1.0

    return df_scaled


def normalize_prof_from_file(df_user_profiles):
    x = df_user_profiles.iloc[:, :-3].values  # returns a numpy array
    print len(x.T)
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x.T)
    print len(df_user_profiles.columns.difference(['user_id', 'rating', 'genre_count']))
    df_scaled = pd.DataFrame(x_scaled.T, columns=df_user_profiles.columns.difference(
        ['user_id', 'rating', 'genre_count']))

    df_scaled['user_id'] = df_user_profiles['user_id'].values
    df_scaled['genre_count'] = map(
        lambda x: x / 13.0, df_user_profiles['genre_count'].values)
    df_scaled['rating'] = 1.0

    return df_scaled


def get_user_profile(user_id, df_rating, data_anime):
    df_anime_genres = pd.get_dummies(
        data_anime['genre'].str.get_dummies(sep=", "))  # creates genre vectors
    df_anime_vector = pd.concat(
        [data_anime['anime_id'], df_anime_genres], axis=1)

    df_user = df_rating.loc[df_rating['user_id'] == user_id]
    df_merged = pd.merge(df_user, df_anime_vector, how='left', left_on='anime_id', right_on='anime_id'
                         ).drop(['anime_id', 'rating'], axis=1)

    avg_genre = df_merged[df_merged.columns.difference(
        ['user_id'])].sum(axis=1)

    # Count only 1's
    df_user_sum = df_merged.apply(
        pd.Series.value_counts).loc[df_merged.index == 1]
    df_user_sum.fillna(0, inplace=True)
    df_user_sum = df_user_sum.apply(func=lambda x: x**2, axis=0)

    df_user_sum['genre_count'] = avg_genre.sum() / float(len(avg_genre))
    df_user_sum['user_id'] = user_id
   # df_user_sum['rating'] = 10.0

    return df_user_sum


def build_user_profiles(user_ids):
    df_user_profiles = pd.DataFrame()

    for id in user_ids:
        u_prof = get_user_profile(id, data_rating, data_anime)
        df_user_profiles = df_user_profiles.append(u_prof, ignore_index=True)

    return df_user_profiles


def build_knn(n, id, rating=False):
    filter_out = train_rating.loc[train_rating['user_id'] == id]['anime_id']
    filter_anime = data_anime.loc[~data_anime['anime_id'].isin(
        set(filter_out))]

    filter_anime_genres = pd.get_dummies(
        filter_anime['genre'].str.get_dummies(sep=", "))  # creates genre vectors
    df_anime_vector = pd.concat(
        [filter_anime['anime_id'], filter_anime_genres], axis=1)  # anime_id + genre vector
    df_anime_vector['genre_count'] = df_anime_vector[df_anime_vector.columns.difference(
        ['anime_id'])].sum(axis=1).apply(lambda x: x / 13.0)
    if rating:
        filter_anime_genres['rating'] = 0
        df_anime_vector['rating'] = filter_anime['rating'].apply(
            lambda x: x / 10.0)
        df_anime_vector.fillna(0, inplace=True)

    return NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(df_anime_vector.iloc[:, 1:])


def get_n_closest_users(norm_profile, n, rating):

    nbrs = build_knn(n, norm_profile.user_id, rating=rating)
    norm_profile = norm_profile.drop('user_id')

    # Get closest neighbours
    distances, indices = nbrs.kneighbors(norm_profile)

    return distances, indices, norm_profile


# In[337]:


profile1 = get_user_profile(1, train_rating, data_anime)
profile2 = get_user_profile(2, train_rating, data_anime)

profiles = pd.DataFrame.append(profile1, profile2)
# print normalize(profiles)

final_normalized = normalize_prof_from_file(final_profiles)
final_normalized.head(5)


# In[344]:

# profiles = build_user_profiles([10203,43202,1300])


usdf = pd.DataFrame()

with open('content_recommendations.csv', 'ab') as file:
    writer = csv.writer(file)
    for idx in range(0, 2000):
        distances, indices, us = get_n_closest_users(
            final_normalized.drop([], axis=1).iloc[idx], 10, True)

        # usdf = usdf.append(final_normalized.iloc[idx], ignore_index=True)
        # test_movies = test_rating.loc[test_rating['user_id']
        #                               == final_normalized.iloc[idx]['user_id']]
        for ind in indices:
            print "-----------------------"

            # print data_anime.loc[ind][['anime_id','genre', 'rating']]
            # print
            # set(data_anime.loc[ind]['anime_id']).intersection(set(test_movies['anime_id']))
            rec = [final_normalized.iloc[idx]['user_id'],
                   list(data_anime.loc[ind]['anime_id'])]
            print rec
            writer.writerow(rec)
