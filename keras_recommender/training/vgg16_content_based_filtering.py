import pandas as pd
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
from keras_recommender.library.cnn import Vgg16ContentBaseFiltering
import os
import pickle
from glob import glob

def main():
    data_dir_path = './data/ml-latest-small'
    poster_dir_path = './data/posters'
    output_dir_path = './data/models'

    np.set_printoptions(threshold=np.nan)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df = pd.read_csv(data_dir_path + '/ratings.csv', sep=',')
    df_id = pd.read_csv(data_dir_path + '/links.csv', sep=',', dtype=object, index_col=0)
    df_movie_names = pd.read_csv(data_dir_path + '/movies.csv', sep=',')
    df = pd.merge(pd.merge(df, df_id, on='movieId'), df_movie_names, on='movieId')

    print(df.head())

    data_file = data_dir_path + '/imdb_id_to_image_dict.data'
    if not os.path.exists(data_file):
        imdb_id_to_image_dict = dict()
        for poster_file in glob(poster_dir_path + '/*.jpg'):  # debug here
            print('Loading img at {}'.format(poster_file))
            img = kimage.load_img(poster_file, target_size=(224, 224))
            img = preprocess_input(np.expand_dims(kimage.img_to_array(img), axis=0))
            imdb_id = poster_file.split('/')[-1].split('.')[0]
            imdb_id_to_image_dict[imdb_id] = img
        pickle.dump(file=open(data_file, 'wb'), obj=imdb_id_to_image_dict)
    else:
        imdb_id_to_image_dict = pickle.load(file=open(data_file, 'rb'))

    recommender = Vgg16ContentBaseFiltering()
    recommender.fit(imdb_id_to_image_dict, model_dir_path=output_dir_path)




if __name__ == '__main__':
    main()