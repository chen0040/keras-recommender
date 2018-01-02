import pandas as pd
from keras_recommender.library.cf import CollaborativeFilteringWithTemporalInformation


def main():
    data_dir_path = './data/ml-latest-small'
    trained_model_dir_path = './models'

    records = pd.read_csv(data_dir_path + '/ratings.csv')
    print(records.describe())

    timestamp_test = records['timestamp']
    user_id_test = records['userId']
    item_id_test = records['movieId']
    rating_test = records['rating']

    cf = CollaborativeFilteringWithTemporalInformation()
    cf.load_model(CollaborativeFilteringWithTemporalInformation.get_config_file_path(trained_model_dir_path),
                  CollaborativeFilteringWithTemporalInformation.get_weight_file_path(trained_model_dir_path))

    predicted_ratings = cf.predict(user_id_test, item_id_test, timestamp_test)
    print(predicted_ratings)
    
    for i in range(20):
        user_id = user_id_test[i]
        item_id = item_id_test[i]
        timestamp = timestamp_test[i]
        rating = rating_test[i]
        predicted_rating = cf.predict_single(user_id, item_id, timestamp)
        print('predicted: ', predicted_rating, ' actual: ', rating)


if __name__ == '__main__':
    main()
