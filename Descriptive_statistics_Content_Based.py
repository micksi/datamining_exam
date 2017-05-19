
# coding: utf-8

# In[2]:

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


# In[3]:

data_rating = pd.io.parsers.read_csv('raw/rating.csv')
#data_rating = data_rating.loc[data_rating['user_id'] != 48766]
data_anime = pd.io.parsers.read_csv('raw/anime.csv')

train_rating = pd.io.parsers.read_csv('omer/rating_train.csv')
test_rating = pd.io.parsers.read_csv('omer/rating_test.csv')
final_profiles = pd.io.parsers.read_csv('raw/user_profiles_final.csv')

print "Animes: "
print data_anime.describe()

print "\nRatings: "
print data_rating.describe()


# In[241]:

data_anime['rating'].plot(kind="hist", bins=40,figsize=(15,4))
plt.show()


# In[3]:

data_rating.groupby('user_id').size().to_frame().sort_values(by=0).plot(kind="density", logx=True, figsize=(15,4))
print data_rating.groupby('user_id').size().to_frame().sort_values(by=0).quantile(0.01)
plt.show()


# In[4]:

print 'Anime rating:'
print data_anime['rating'].describe(include='all')

for t in ['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music']:
    print '\nRating: ' + t
    print data_anime.loc[data_anime['type'] == t]['rating'].describe(include='all')
    data_anime.loc[data_anime['type'] == t]['rating'].plot(kind="density", figsize=(15,8))

L=plt.legend()
for i, t in enumerate(['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music']):
    L.get_texts()[i].set_text(t)

plt.show()


# In[5]:

_genre = data_anime['genre']
_genre_list = []
genre_count = []
unique_genre = []
for g in _genre:
    try:
        gs = g.split(',')
        if len(gs) == 1:
            unique_genre.extend(gs)
        _genre_list.extend(map(lambda s: s.strip(), gs))
        genre_count.append(gs)
    except:
        pass
    
#print genre_count / len(data_anime) 

print sorted(set(unique_genre))
print len(sorted(set(unique_genre)))

print len(data_anime) 
print len(sorted(set(_genre_list)))
print sorted(set(_genre_list))
print
print set(_genre_list) - set(unique_genre)

pd.DataFrame(map(lambda x: len(x), genre_count)).plot(kind='hist', bins=15)
plt.show()


# In[6]:

genre_count = pd.DataFrame(_genre_list).groupby(0)
genre_count.size().sort_values(ascending=False).plot(kind="bar", width=0.9, figsize=(15,4))
plt.show()


# In[7]:

members = data_anime['members'].cumsum()

data_anime['members'].quantile(np.arange(0.0, 1.0, 0.01)).plot(kind="line")
plt.show()


# In[8]:

data_anime['members'].plot(kind="box", logy=True)
plt.show()


# In[9]:

movies = data_rating.groupby('user_id').size().to_frame().sort_values(by=0)
movies.loc[movies[0] < 314].describe()
movies.loc[movies[0] < 314].plot(kind="hist", bins=313)
plt.show()


# In[4]:

def normalize(df_user_profiles):
    x = df_user_profiles.iloc[:,1:].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    
    x_scaled = min_max_scaler.fit_transform(x)
    
    df_scaled = pd.DataFrame(x_scaled, columns=df_user_profiles.columns.difference(['user_id','rating','genre']))
    
    df_scaled['user_id'] = df_user_profiles['user_id'].values
    df_scaled['genre_count'] = map(lambda x: x /10.0, df_user_profiles['genre_count'].values)
    #df_scaled['rating'] = 1.0
    
    return df_scaled

def normalize_prof_from_file(df_user_profiles):
    x = df_user_profiles.iloc[:,:-3].values #returns a numpy array
    print len(x.T)
    min_max_scaler = preprocessing.MinMaxScaler()
    
    x_scaled = min_max_scaler.fit_transform(x.T)
    print len(df_user_profiles.columns.difference(['user_id','rating','genre_count']))
    df_scaled = pd.DataFrame(x_scaled.T, columns=df_user_profiles.columns.difference(['user_id','rating','genre_count']))
    
    df_scaled['user_id'] = df_user_profiles['user_id'].values
    df_scaled['genre_count'] = map(lambda x: x /13.0, df_user_profiles['genre_count'].values)
    df_scaled['rating'] = 1.0
    
    return df_scaled

def get_user_profile(user_id, df_rating, data_anime):
    df_anime_genres = pd.get_dummies(data_anime['genre'].str.get_dummies(sep=", ")) # creates genre vectors
    df_anime_vector = pd.concat([data_anime['anime_id'], df_anime_genres], axis=1)
    
    df_user = df_rating.loc[df_rating['user_id'] == user_id]
    df_merged = pd.merge(df_user, df_anime_vector, how='left', left_on='anime_id', right_on='anime_id' 
                        ).drop(['anime_id', 'rating'], axis=1)

    
    avg_genre = df_merged[df_merged.columns.difference(['user_id'])].sum(axis=1)
    
    # Count only 1's
    df_user_sum = df_merged.apply(pd.Series.value_counts).loc[df_merged.index == 1]
    df_user_sum.fillna(0, inplace = True)
    df_user_sum = df_user_sum.apply(func=lambda x: x**2,axis=0)

    df_user_sum['genre_count'] = avg_genre.sum() / float(len(avg_genre))
    df_user_sum['user_id'] = user_id
   # df_user_sum['rating'] = 10.0

    return df_user_sum

def build_user_profiles(user_ids):
    df_user_profiles = pd.DataFrame()

    for id in user_ids:
        u_prof = get_user_profile(id, data_rating, data_anime)
        df_user_profiles = df_user_profiles.append(u_prof, ignore_index = True)
    
    return df_user_profiles

def build_knn(n, id, rating=False):
    filter_out = train_rating.loc[train_rating['user_id'] == id]['anime_id']
    filter_anime = data_anime.loc[~data_anime['anime_id'].isin(set(filter_out))]
    
    filter_anime_genres = pd.get_dummies(filter_anime['genre'].str.get_dummies(sep=", ")) # creates genre vectors
    df_anime_vector = pd.concat([filter_anime['anime_id'], filter_anime_genres], axis=1) # anime_id + genre vector
    df_anime_vector['genre_count'] =  df_anime_vector[df_anime_vector.columns.difference(['anime_id'])].sum(axis=1).apply(lambda x: x / 13.0)
    if rating:
        filter_anime_genres['rating'] = 0
        df_anime_vector['rating'] =  filter_anime['rating'].apply(lambda x: x / 10.0)
        df_anime_vector.fillna(0, inplace = True)

    return NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(df_anime_vector.iloc[:,1:])

def get_n_closest_users(norm_profile, n, rating):
    
    nbrs = build_knn(n, norm_profile.user_id, rating=rating)
    norm_profile = norm_profile.drop('user_id')

    # Get closest neighbours
    distances, indices = nbrs.kneighbors(norm_profile)
    
    return distances, indices, norm_profile
    


# In[11]:

profile1 = get_user_profile(1, train_rating, data_anime)
profile2 = get_user_profile(2, train_rating, data_anime)

profiles = pd.DataFrame.append(profile1,profile2)
# print normalize(profiles)

final_normalized = normalize_prof_from_file(final_profiles)
final_normalized.head(5)


# In[13]:

data_rating.loc[data_rating['user_id'] == 102]


# In[10]:

# profiles = build_user_profiles([10203,43202,1300])


usdf = pd.DataFrame()

with open('content_recommendations', 'ab') as file:
    writer = csv.writer(file)
    for idx in [5,6,7,8,14,17,21,23,25,26,27]:    
        distances, indices, us = get_n_closest_users(final_normalized.drop([], axis=1).iloc[idx], 10, True)

        usdf = usdf.append(final_normalized.iloc[idx], ignore_index=True)
        test_movies = test_rating.loc[test_rating['user_id'] == final_normalized.iloc[idx]['user_id']]
        for ind in indices:
            print "-----------------------"
            print final_normalized.iloc[idx]['user_id']
            # print data_anime.loc[ind][['anime_id','genre', 'rating']]
            print len(data_anime.loc[ind]['anime_id'])
            print set(data_anime.loc[ind]['anime_id']).intersection(set(test_movies['anime_id']))
            writer.writerow([final_normalized.iloc[idx]['user_id'], ])

