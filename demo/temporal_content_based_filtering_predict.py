import pandas as pd
from sklearn.model_selection import train_test_split
from keras_recommender.library.content_based_filtering import TemporalContentBasedFiltering


def main():
    data_dir_path = './data/ml-latest-small'
    trained_model_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    timestamp_test = records['timestamp']
    item_id_test = records['movieId']
    rating_test = records['rating']

    max_item_id = records['movieId'].max()

    print(timestamp_test.head())

    config = dict()
    config['max_item_id'] = max_item_id

    cf = TemporalContentBasedFiltering()
    cf.load_model(TemporalContentBasedFiltering.get_config_file_path(trained_model_dir_path),
                  TemporalContentBasedFiltering.get_weight_file_path(trained_model_dir_path))

    predicted_ratings = cf.predict(item_id_test, timestamp_test)
    print(predicted_ratings)

    for i in range(20):
        date = timestamp_test[i]
        item_id = item_id_test[i]
        rating = rating_test[i]
        predicted_rating = cf.predict_single(item_id, date)
        print('predicted: ', predicted_rating, ' actual: ', rating)


if __name__ == '__main__':
    main()
