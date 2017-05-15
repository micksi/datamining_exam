import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.max_colwidth', 300)
file_name = "anime.csv"
file_name_rating = 'rating_train.csv'
file_name_test = 'rating_test.csv'

def normalize(df_user_profiles):
    x = df_user_profiles.iloc[:,1:].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.T)
    df_user_profiles = pd.concat([df_user_profiles['user_id'], pd.DataFrame(x_scaled.T, columns=df_user_profiles.columns[1:])], axis=1)
    return df_user_profiles

def get_user_profile(user_id, df_rating, df_a_fatures):
    df_user = df_rating.loc[df_rating['user_id'] == user_id]
    df_merged = pd.merge(df_user, df_a_fatures, how='left', left_on='anime_id', right_on='anime_id').drop(['anime_id', 'rating'], axis=1)
    
    avg_genre = df_merged.apply(np.sum, axis=1).cumsum()
    print avg_genre
    # Count only 1's
    df_user_sum = df_merged.apply(pd.Series.value_counts).loc[df_merged.index == 1]
    df_user_sum.fillna(0, inplace = True)
    df_user_sum = df_user_sum.apply(func=lambda x: x**2,axis=0)
    df_user_sum.user_id = user_id
    print df_user_sum
    return df_user_sum

def print_animes_by_indices(indices, df):
    for i in indices:
        print df.loc[i]
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
# print "Norm-ized"
# print df_user_prof_norm[ df_user_prof_norm['user_id'] == 1 ].T

# User 
user_id = 1
user_animes = df_rating[df_rating['user_id'] == user_id] 
user_animes = user_animes['anime_id'].tolist() # animes watched by the user
# print user_animes[121]
# user_animes = user_animes[:121]


# Remove the animes watched by the user
# df_anime_vector_foruser = df_anime_vector[~df_anime_vector['anime_id'].isin(user_animes)]
# print "1421"
# print df_anime_vector_foruser.loc['1421']
# # Feed the animes
# print df_anime_vector.loc[df_anime_vector['anime_id'] == 22877].values
# print df_anime_vector.loc[df_anime_vector['anime_id'] == 30544].values

nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(df_anime_vector.iloc[:,1:])

user_prof = df_user_prof_norm[df_user_prof_norm['user_id'] == user_id]
user_prof = user_prof.drop('user_id', axis=1)
# Get closest neighbours
distances, indices = nbrs.kneighbors(user_prof)

# def get_index_from_name(name):
#     return df_animes[df_animes["name"]==name].index.tolist()[0]

# print "get_index_from_name('Naruto')"
# print get_index_from_name("Naruto")

# print "indices:"
# print indices
print "Our recommendations: "
# print_animes_by_indices(indices.tolist(), df_anime_vector)

def anime_ids_by_indices(indeces, df):
    anime_ids = []
    print "print:"
    for i in indices:
        # print df.iloc[i,:1]['anime_id'].tolist()
         anime_ids.append(df.loc[i]['anime_id'].tolist()[0])
    # return anime_ids
    print anime_ids

#
# df_tests = pd.read_csv(file_name_test)
# test_animes = df_tests['anime_id'].tolist()
# set(b1).intersection(b2)

print "Distances: "
print distances

# print "anime ids:"
# recomm_anime_ids = anime_ids_by_indices(indices, df_anime_vector_foruser)
# file_rating_test = "rating_test.csv"
# df_rating_test = pd.read_csv(file_rating_test)
# test_anime_ids = df_rating_test[ df_rating_test['user_id'] == user_id ]['anime_id'].tolist()
# print "intersection:"
# print recomm_anime_ids
# set(test_anime_ids).intersection(set(recomm_anime_ids))


# print "User profile (non-normalized):"
# print df_user_profiles[df_user_profiles['user_id'] == user_id].T

# file_rating_test = "rating_test.csv"
# df_rating_test = pd.read_csv(file_rating_test)
# print df_rating_test[ df_rating_test['user_id'] == user_id ]

# print "print_animes_by_indices"
# print print_animes_by_indices([841], df_animes)