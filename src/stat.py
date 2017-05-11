import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd


def main():
    '''
    anime_id
    name
    genre   contains value nan for non specific genre
    type
    episodes
    rating
    members

    '''
    matplotlib.style.use('ggplot')
    pd.options.display.float_format = '{:20,.2f}'.format

    data_rating = pd.io.parsers.read_csv('../raw/rating.csv')
    data_movies = pd.io.parsers.read_csv('../raw/anime.csv')

    # ratingVal = datarating.loc[(datarating['rating']
    #                             > 5) | (datarating['rating'])]['user_id'].value_counts().to_frame()
    # print ratingVal.to_frame().columns
    # print ratingVal.describe()

    # graphs of normal distributions
    show_users_and_movies(data_rating)
    show_movie_ratings(data_movies)

    # number_of_unique_genres(data)
    # number_of_types(data)
    # rating(data, types=['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music'])
    # members(data, plot=True)

    # print '----'
    # print data.loc[(data['type'] == 'OVA')]['episodes'].value_counts()

    plt.show()


def show_movie_ratings(movies):
    plt.figure()
    movies['rating'].plot(kind="hist", bins=40)


def show_users_and_movies(rating):
    describe = rating['user_id'].value_counts().describe()
    print describe['mean']
    rating_user = rating.groupby('user_id')
    users_movies = rating_user.size().to_frame().sort_values(by=0)[:-1]
    plt.figure()
    users_movies.plot(kind="hist", bins=100)


def number_of_types(data, plot=False):
    print "\nDescribe: "
    typeDescription = data['type'].describe(include="all")
    print typeDescription
    print "\nType count: "
    print data[data.columns[3]].value_counts()


def members(data, plot=False):
    print '\nMembers:'
    print data['members'].describe(include='all')

    if plot:
        print "plotting..."
        members = data['members'].cumsum()

        data['members'].quantile(np.arange(0.0, 1.0, 0.01)).plot(
            kind="line", ax=axes[0])
        data['members'].plot(kind="box", logy=True, ax=axes[1])


def rating(data, types=None, plot=False):
    print '\nRating:'
    print data['rating'].describe(include='all')

    if types:
        for t in types:
            print '\nRating: ' + t
            print data.loc[data['type'] == t]['rating'].describe(include='all')


def number_of_unique_genres(data, plot=False):
    _genre = data['genre']
    _genre_list = []
    for g in _genre:
        try:
            _genre_list.extend(map(lambda s: s.strip(), g.split(',')))
        except:
            pass
    print len(sorted(set(_genre_list)))
    print sorted(set(_genre_list))


if __name__ == "__main__":
    main()
