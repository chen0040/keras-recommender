from sklearn.model_selection import train_test_split
import pandas as pd
from keras_recommender.library.cf import CollaborativeFilteringV1

def main():
    data_dir_path = './data/ml-latest-small'
    output_dir_path = './models'

    all_ratings = pd.read_csv(data_dir_path + '/ratings.csv')
    all_ratings['rating'].describe()

    ratings_train, ratings_test = train_test_split(all_ratings, test_size=0.2, random_state=0)

    user_id_train = ratings_train['user_id']
    item_id_train = ratings_train['item_id']
    rating_train = ratings_train['rating']

    user_id_test = ratings_test['user_id']
    item_id_test = ratings_test['item_id']
    rating_test = ratings_test['rating']

    max_user_id = all_ratings['user_id'].max()
    max_item_id = all_ratings['item_id'].max()

    config = dict()
    config['max_user_id'] = max_user_id
    config['max_item_id'] = max_item_id

    cf = CollaborativeFilteringV1()
    cf.fit(config=config, user_id_train=user_id_train,
           item_id_train=item_id_train,
           rating_train=rating_train,
           model_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
