import os
import random

import pandas as pd
import requests
import wget
from bs4 import BeautifulSoup


def download_poster(movie_id):
    base_url = 'http://www.imdb.com/title/tt{}/'.format(movie_id)
    print(base_url)
    return BeautifulSoup(requests.get(base_url).content, 'lxml').find('div', {'class': 'poster'}).find('img').attrs[
        'src']


def download_posters(data_dir_path=None):
    if data_dir_path is None:
        data_dir_path = '../training/data'

    df_id = pd.read_csv(data_dir_path + '/ml-latest-small/links.csv', sep=',', dtype=object)

    movies = []
    for idx, imdb_id in enumerate(df_id['imdbId']):
        movies.append(imdb_id)

    total_movies = len(movies)

    print('total movies: ', total_movies)

    poster_path = data_dir_path + '/posters'

    # random.shuffle(movies)
    for movie_id in movies:
        out = os.path.join(poster_path, str(movie_id) + '.jpg')
        if not os.path.exists(out):
            try:
                target = download_poster(movie_id)
                print('Download img from [{0}] to [{1}].'.format(target, out))
                wget.download(url=target, out=out, bar=None)
            except AttributeError:
                print('IMDB does not have picture for this movie: ', movie_id)
        else:
            print('Image already exists {0}'.format(out))


def main():
    download_posters()


if __name__ == '__main__':
    main()

