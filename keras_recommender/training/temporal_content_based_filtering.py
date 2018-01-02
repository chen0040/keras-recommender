import pandas as pd
from sklearn.model_selection import train_test_split
from keras_recommender.library.content_based_filtering import TemporalContentBasedFiltering


def main():
    data_dir_path = './data/ml-latest-small'
    output_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    ratings_train, ratings_test = train_test_split(records, test_size=0.2, random_state=0)

    date_train = ratings_train['timestamp']
    item_id_train = ratings_train['movieId']
    rating_train = ratings_train['rating']

    date_test = ratings_test['timestamp']
    item_id_test = ratings_test['movieId']
    rating_test = ratings_test['rating']

    max_item_id = records['movieId'].max()

    config = dict()
    config['max_item_id'] = max_item_id

    cf = TemporalContentBasedFiltering()
    history = cf.fit(config=config, date_train=date_train,
                     item_id_train=item_id_train,
                     rating_train=rating_train,
                     model_dir_path=output_dir_path)

    metrics = cf.evaluate(user_id_test=date_test,
                          item_id_test=item_id_test,
                          rating_test=rating_test)


if __name__ == '__main__':
    main()
