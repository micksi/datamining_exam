
# coding: utf-8

# In[14]:

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import scipy
import csv

matplotlib.style.use('ggplot')
pd.options.display.float_format = '{:20,.2f}'.format
pd.set_option('display.max_columns', 50)


# In[4]:

test_rating = pd.io.parsers.read_csv('raw/rating_test.csv')


# In[5]:

content = pd.io.parsers.read_csv('content_recommendations.csv')
content['recs'] = content['recs'].apply(lambda s: map(lambda _s: _s.replace("'","").strip(), s[1:-1].split(',')) )


# In[81]:

content_no_rating = pd.io.parsers.read_csv('content_recommendations_no_rating.csv')
content_no_rating['recs'] = content_no_rating['recs'].apply(lambda s: map(lambda _s: _s.replace("'","").strip(), s[1:-1].split(',')) )


# In[6]:

collaborative = pd.io.parsers.read_csv('collaborative_recommendations.csv')
collaborative['recs'] = collaborative['recs'].apply(lambda s: map(lambda _s: _s.replace("'","").strip(), s[1:-1].split(',')) )
collaborative['size'] = collaborative['recs'].apply(lambda l: len(l))
collaborative = collaborative.loc[collaborative['size'] == 10]


# In[117]:

user_ids = list(collaborative['user_id'].unique())
user_results = pd.DataFrame()

for user_id in user_ids:
    # Get the results of the collaborative filtering recommendations
    recommendations_cf = collaborative[collaborative['user_id']==user_id]['recs'].tolist()[0]
    recommendations_cf = set(map(lambda x: int(x), recommendations_cf))
    
    # print(recommendations_cf)
    # Get the results of the content based recommendations
    recommendations_cb = content[content['user_id']==user_id]['recs'].tolist()[0]
    recommendations_cb = set(map(lambda x: int(x), recommendations_cb))
    
    recommendations_cb_no_rating = content_no_rating[content_no_rating['user_id']==user_id]['recs'].tolist()[0]
    recommendations_cb_no_rating = set(map(lambda x: int(x), recommendations_cb_no_rating))
    
    # Get the test sample for the user
    watched_animes = test_rating['anime_id'][test_rating['user_id'] == user_id].astype(int).tolist()
    watched_animes = set(watched_animes)
    
    
    union = recommendations_cf.union(recommendations_cb)
    inter = recommendations_cf.intersection(recommendations_cb)
    
    # Get the denominator for the collaborative filtering and content based recommendations
    if len(watched_animes) > 10:
        denom = 10
    else:
        denom = len(watched_animes)
    
    if user_id == 32:
        print watched_animes
        print sorted(recommendations_cb)
        print sorted(recommendations_cf)
    
    # Get the 4 results for each user: number of matches for cf, cb, union and intersection
    res_cf = len(watched_animes.intersection(recommendations_cf)) #* 1.0 / denom
    res_cb = len(recommendations_cb.intersection(watched_animes)) #* 1.0 / denom
    res_cb_no_rating = len(recommendations_cb_no_rating.intersection(watched_animes)) #* 1.0 / denom
    res_union = len(union.intersection(watched_animes)) * 1.0 / min(len(union), len(watched_animes))
    if min(len(inter), len(watched_animes)) != 0:  
        res_intersection = len(inter.intersection(watched_animes)) * 1.0 / min(len(inter), len(watched_animes))
    else: 
        res_intersection = 0.0
        
    user_results = user_results.append({
        'user_id': user_id, 
        'res_cb': res_cb, 
        'res_cb_no_rating': res_cb_no_rating,  
        'res_cf':res_cf, 
        'res_union':res_union, 
        'res_inter':res_intersection, 
        'denom':denom, 
        'watched_animes': len(watched_animes)
    }, ignore_index=True)



# In[109]:

test_rating['anime_id'][test_rating['user_id'] == 377]


# In[114]:

user_results.loc[user_results['res_union'] > 0.3]


# In[57]:

user_results['res_union'].sum() / len(user_results['res_union'])


# In[125]:

user_results.describe()


# In[134]:

data=user_results['res_cf'].sort_values().plot(kind="hist", bins=10, label="Collaborative")
#y,binEdges=np.histogram(data,bins=10)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'-')

data=user_results['res_cb'].sort_values().plot(kind="hist", bins=10, label="Content", figsize=(10,7))
#y,binEdges=np.histogram(data,bins=10)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.plot(bincenters,y,'-')

plt.xlabel('number of accurate recommendations')
plt.ylabel('number of users')
plt.legend()
plt.xticks(np.arange(0,10))
plt.show()


# In[141]:

data=user_results['res_union'].sort_values()
y,binEdges=np.histogram(data,bins=10)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.figure(figsize=(10,7))
plt.plot(bincenters,y,'-')

plt.xlabel('recommendation accuracy')
plt.ylabel('number of users')
plt.show()

