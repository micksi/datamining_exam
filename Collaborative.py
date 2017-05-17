import sys
import pandas as pd
import numpy as np
import csv
import time
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import Orange


def get_user_profile(user_id, df_rating, df_a_fatures):

    # To be used only if the user profiles file is not already created
    df_user = df_rating.loc[df_rating['user_id'] == user_id]
    df_merged = pd.merge(df_user, df_a_fatures, how='left', left_on='anime_id', right_on='anime_id').drop(['anime_id', 'rating'], axis=1)

    avg_genre = df_merged[df_merged.columns.difference(['user_id', 'anime_id', 'rating'])].sum(axis=1)

    # Count only 1's
    df_user_sum = df_merged.sum(axis=0)
    df_user_sum.user_id = user_id
    df_user_sum['rating'] = 10.0
    df_user_sum['genre_count'] = avg_genre.sum() / float(len(avg_genre))

    return df_user_sum


def get_user_profiles(df_animes_vector, df_rating, n_users=50):

    # To be used only if the user profiles file is not already created

    # first n_users
    users = list(df_rating['user_id'].unique())[:n_users]

    # Create user profiles:
    df_user_profiles = pd.DataFrame()
    i = 0
    for u in users:
        u_prof = get_user_profile(u, df_rating, df_animes_vector)
        df_user_profiles = df_user_profiles.append(u_prof, ignore_index=True)
        i = i + 1
        if i % 100 == 0:
            print("Completed users:", i)

    return df_user_profiles


def normalize(df_user_profiles):
    x = df_user_profiles.iloc[:, 0:-2].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x.T)

    df_scaled = pd.DataFrame(x_scaled.T, columns=df_user_profiles.columns.difference(['user_id', 'rating', 'genre']))

    df_scaled['user_id'] = df_user_profiles['user_id'].values
    df_scaled['genre_count'] = map(lambda x: x / 10.0, df_user_profiles['genre_count'].values)
    df_scaled['rating'] = 1.0

    return df_scaled


def get_userids_by_indices(indices, df_user_prof_norm):
    users = []
    for i in indices:
        uid = df_user_prof_norm.loc[i]['user_id']
        users.append(uid)
    return users


def get_collaborative_recommendations_per_user(user_id, k, df_user_prof_norm, df_rating):

    # find closest k user profiles
    start_kNN_duration = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(df_user_prof_norm.drop(['user_id', 'rating', 'genre_count'], axis=1))
    user_prof = df_user_prof_norm[df_user_prof_norm['user_id'] == user_id]
    user_prof = user_prof.drop(['user_id', 'rating', 'genre_count'], axis=1)

    # Get closest neighbours
    distances, indices = nbrs.kneighbors(user_prof)
    kNN_duration = time.time() - start_kNN_duration
    print("Closest neighbours identified in %s seconds!" % kNN_duration)

    # get user_ids
    uids = get_userids_by_indices(indices[0], df_user_prof_norm)

    # ------------------------------------------------------------
    u_animes = []
    for uid in uids:
        u_animes.append(df_rating[df_rating['user_id'] == uid]['anime_id'].tolist())
    with open('anime_trans.basket', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(u_animes)
    # ------------------------------------------------------------

    # Get all training transactions

    data = Orange.data.Table("anime_trans.basket") #Orange.data.Table("anime_trans.basket")

    # This is the user we would like to recommend something for
    target_user = data[0]
    target_user_animes = data[0].get_metas(str).keys()

    # Drop the user's data from the transactions list
    data = data.get_items(range(1, len(data)))

    # Generate recommendation rules
    starting_time = time.time()
    support_threshold = 0.5
    confidence_threshold = 0.8
    rulesOK = False
    while rulesOK is False:
        try:
            rules = Orange.associate.AssociationRulesSparseInducer(data, support=support_threshold, confidence=confidence_threshold,
                                                                   max_item_sets=100000)
            print(len(rules))
            # Test for the number of generated rules
            if len(rules) > 100000:
                print(support_threshold, confidence_threshold)
                if confidence_threshold == 1:
                    support_threshold += 0.1
                else:
                    confidence_threshold += 0.1
            else:
                rulesOK = True
        except:
            print(support_threshold, confidence_threshold)
            if confidence_threshold == 1:
                support_threshold += 0.1
            else:
                confidence_threshold += 0.1

    # print "%4s\t %4s  %s %s" % ("AnimeId", "Lift", "Support", "Conf")
    recommendations = {}
    for r in rules:

        # Compare the generated rules with a specific instance from the transactions list
        if(r.n_right == 1):
            recommendation = str(r.right.get_metas(str).keys()[0])
            if recommendation not in target_user_animes:
                # if r.applies_left(target_user):
                try:
                    recommendations[r.n_left].append(r)
                except:
                    recommendations[r.n_left] = []
                    recommendations[r.n_left].append(r)
                    # print "%4.2f %4.4f %s %s" % (r.support, r.confidence, r, r.lift)

    print("We found %s potential rules! Let's check them out!" % len(rules))
    user_recommendations = []
    for i, r in recommendations.iteritems():
        recommendations[i].sort(key=lambda x: (x.lift, x.support, x.confidence), reverse=True)

    for recommendation_length in sorted(recommendations.keys(), reverse=True):
        if len(user_recommendations) == 10:
            break
        for recommendation in recommendations[recommendation_length]:
            anime_id = str(recommendation.right.get_metas(str).keys()[0])
    #         print recommendation
    #         print anime_id, "\t", recommendation.lift, recommendation.support, recommendation.confidence
            if anime_id not in user_recommendations:
                user_recommendations.append(anime_id)
            if len(user_recommendations) == 10:
                break
    duration = time.time() - starting_time
    print("Rules found in %s seconds!" % duration)
    return user_recommendations
    # Orange.associate.AssociationRulesSparseInducer.get_itemsets(rules)

# =================== MAIN =========================================
# Read the user profiles from the user_profiles_final.csv file


user_profiles = "raw/user_profiles_final.csv"
df_user_profiles = pd.read_csv(user_profiles)

file_rating = "raw/rating_train.csv"
df_rating = pd.read_csv(file_rating)

users_ids = list(df_user_profiles['user_id'].unique())
df_user_prof_norm = normalize(df_user_profiles)

recommendations = {}

with open('collaborative.csv', 'ab') as csv_file:
    writer = csv.writer(csv_file)
    for i in users_ids[81:83]:
        print ("Results for user %4d\t " % (i))
        rec = get_collaborative_recommendations_per_user(user_id=i, k=11, df_user_prof_norm=df_user_prof_norm, df_rating=df_rating)
        recommendations[i] = rec
        writer.writerow([i, recommendations[i]])
