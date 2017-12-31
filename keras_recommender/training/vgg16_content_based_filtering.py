import pandas as pd
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage

def main():
    data_dir_path = './data/ml-latest-small'

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


if __name__ == '__main__':
    main()