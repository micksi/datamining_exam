import pandas as pd
import matplotlib as mpl


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
    pd.options.display.float_format = '{:20,.2f}'.format
    data = pd.io.parsers.read_csv('../raw/anime.csv')

    # number_of_unique_genres(data)
    number_of_types(data)
    rating(data, types=['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music'])
    members(data)


def number_of_types(data, plot=False):
    print "\nDescribe: "
    print data['type'].describe(include="all")
    print "\nType count: "
    print data[data.columns[3]].value_counts()


def members(data, plot=False):
    print '\nMembers:'
    print data['members'].describe(include='all')


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
