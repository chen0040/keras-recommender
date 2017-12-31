import os
import random

import pandas as pd
import requests
import wget
from bs4 import BeautifulSoup


def get_poster(movie_id):
    base_url = 'http://www.imdb.com/title/tt{}/'.format(movie_id)
    print(base_url)
    return BeautifulSoup(requests.get(base_url).content, 'lxml').find('div', {'class': 'poster'}).find('img').attrs[
        'src']


df_id = pd.read_csv('ml-latest-small/links.csv', sep=',')

idx_to_movie = {}
for row in df_id.itertuples():
    idx_to_movie[row[1] - 1] = row[2]

total_movies = 9000

movies = [0] * total_movies
for i in range(len(movies)):
    if i in idx_to_movie.keys() and len(str(idx_to_movie[i])) == 6:
        movies[i] = (idx_to_movie[i])
movies = list(filter(lambda imdb: imdb != 0, movies))
total_movies = len(movies)

URL = [0] * total_movies
IMDB = [0] * total_movies
URL_IMDB = {'url': [], 'imdb': []}

poster_path = 'posters'

random.shuffle(movies)
for movie_id in movies:
    out = os.path.join(poster_path, str(movie_id) + '.jpg')
    if not os.path.exists(out):
        try:
            target = get_poster(movie_id)
            print('Download img from [{0}] to [{1}].'.format(target, out))
            wget.download(url=target, out=out, bar=None)
        except AttributeError:
            pass  # IMDB does not have picture for this movie.
    else:
        print('Image already exists {0}'.format(out))
