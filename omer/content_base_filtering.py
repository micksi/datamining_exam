import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.max_colwidth', 300)
file_name = "anime.csv"
file_name_rating = 'rating.csv'

def normalize(df_user_profiles):
    x = df_user_profiles.iloc[:,1:].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.T)
    df_user_profiles = pd.concat([df_user_profiles['user_id'], pd.DataFrame(x_scaled.T, columns=df_user_profiles.columns[1:])], axis=1)
    return df_user_profiles

def get_user_profile(user_id, df_rating, df_a_fatures):
    df_user = df_rating.loc[df_rating['user_id'] == user_id]
    df_merged = pd.merge(df_user, df_a_fatures, how='left', left_on='anime_id', right_on='anime_id').drop(['anime_id', 'rating'], axis=1)
    
    # Count only 1's
    df_user_sum = df_merged.apply(pd.Series.value_counts).loc[df_merged.index == 1]
    df_user_sum.fillna(0, inplace = True)
    df_user_sum.user_id = user_id
    return df_user_sum

def print_animes_by_indices(indices):
    for i in indices:
        print df_animes.iloc[i,:3]
#

df_rating = pd.read_csv(file_name_rating)
df_animes = pd.read_csv(file_name)
df_anime_genres = pd.get_dummies(df_animes['genre'].str.get_dummies(sep=", ")) # creates genre vectors
df_anime_vector = pd.concat([df_animes['anime_id'], df_anime_genres], axis=1) # anime_id + genre vector

# first 10 users
users = list(df_rating['user_id'].unique())[:10] 

# Create user profiles:
df_user_profiles = pd.DataFrame()
for u in users:
    u_prof = get_user_profile(u, df_rating, df_anime_vector)
    df_user_profiles = df_user_profiles.append(u_prof, ignore_index = True)
    # ??? form user profile from 80% of the wathced animes

# Normalize user profile
df_user_prof_norm = normalize(df_user_profiles)

# User 
user_id = 1
user_animes = df_rating[df_rating['user_id'] == user_id] 
user_animes = user_animes['anime_id'].tolist() # animes watched by the user

# Remove the animes watched by the user
df_anime_vector_foruser = df_anime_vector[~df_anime_vector['anime_id'].isin(user_animes)]

# Feed the animes
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(df_anime_vector_foruser.iloc[:,1:])

user_prof = df_user_prof_norm[df_user_prof_norm['user_id'] == user_id]
user_prof = user_prof.drop('user_id', axis=1)
# Get closest neighbours
distances, indices = nbrs.kneighbors(user_prof)

print "Our recommendations: "
print_animes_by_indices(indices.tolist())

print "User profile (non-normalized):"
print df_user_profiles[df_user_profiles['user_id'] == user_id].T
