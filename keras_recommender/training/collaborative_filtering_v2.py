from sklearn.model_selection import train_test_split
import pandas as pd
from keras_recommender.library.cf import CollaborativeFilteringV2


def main():
    data_dir_path = './data/ml-latest-small'
    output_dir_path = './models'

    all_ratings = pd.read_csv(data_dir_path + '/ratings.csv')
    print(all_ratings.describe())

    ratings_train, ratings_test = train_test_split(all_ratings, test_size=0.2, random_state=0)

    user_id_train = ratings_train['userId']
    item_id_train = ratings_train['movieId']
    rating_train = ratings_train['rating']

    user_id_test = ratings_test['userId']
    item_id_test = ratings_test['movieId']
    rating_test = ratings_test['rating']

    max_user_id = all_ratings['userId'].max()
    max_item_id = all_ratings['movieId'].max()

    config = dict()
    config['max_user_id'] = max_user_id
    config['max_item_id'] = max_item_id

    cf = CollaborativeFilteringV2()
    cf.fit(config=config, user_id_train=user_id_train,
           item_id_train=item_id_train,
           rating_train=rating_train,
           model_dir_path=output_dir_path)

    metrics = cf.evaluate(user_id_test=user_id_test,
                          item_id_test=item_id_test,
                          rating_test=rating_test)


if __name__ == '__main__':
    main()
