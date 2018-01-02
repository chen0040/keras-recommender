# keras-recommender

Recommender built using keras

The dataset is taken from  [ml-latest-small (MovieLens)](https://grouplens.org/datasets/movielens/)

The trained models an be found in keras_recommender/training/models

# Deep Learning Models

### Collaborative Filtering Models

* Collaborative Filtering V1: hidden factor analysis implementation of CF
    * training: keras_recommender/training/collaborative_filtering_v1.py
    * predicting: keras_recommender/training/collaborative_filtering_v1_predict.py
* Collaborative Filtering V2: CF with feedforward dense layer
    * training: keras_recommender/training/collaborative_filtering_v2.py
    * predicting: keras_recommender/training/collaborative_filtering_v2_predict.py
    
### Content-based Filtering Models

* CNN Content-Based Filtering: Use VGG16 for image simlarity on content-based filtering
    * training: keras_recommender/training/vgg16_content_based_filtering.py
    
* Item-based Content-Based Filtering: Use timestamp information and item on content-based filtering
    * trainng: keras_recommender/training/temporal_content_based_filtering.py
    
# Usage

The following code samples provide an illustration on both training and prediction using a deep 
learning model in the keras_recommender/library. Other deep learning models follow the similar
training and prediction patterns.

### Train CF model

To train a CF model, say CollaborativeFilteringV1, run the following commands:

```bash
pip install requirements.txt

cd keras_recommender/training
python collaborative_filtering_v1.py 
```

The training code in collaborative_filtering_v1.py is quite straightforward and illustrated below:

```python
from sklearn.model_selection import train_test_split
import pandas as pd
from keras_recommender.library.cf import CollaborativeFilteringV1

data_dir_path = './data/ml-latest-small' # refers to keras_recommender/training/data/ml-latest-small folder
trained_model_dir_path = './models' # refers to keras_recommender/training/models folder

records = pd.read_csv(data_dir_path + '/ratings.csv')
print(records.describe())

ratings_train, ratings_test = train_test_split(records, test_size=0.2, random_state=0)

user_id_train = ratings_train['userId']
item_id_train = ratings_train['movieId']
rating_train = ratings_train['rating']

user_id_test = ratings_test['userId']
item_id_test = ratings_test['movieId']
rating_test = ratings_test['rating']

max_user_id = records['userId'].max()
max_item_id = records['movieId'].max()

config = dict()
config['max_user_id'] = max_user_id
config['max_item_id'] = max_item_id

cf = CollaborativeFilteringV1()
history = cf.fit(config=config, user_id_train=user_id_train,
                 item_id_train=item_id_train,
                 rating_train=rating_train,
                 model_dir_path=trained_model_dir_path)

metrics = cf.evaluate(user_id_test=user_id_test,
                      item_id_test=item_id_test,
                      rating_test=rating_test)

```

After the training is completed, the trained models will be saved as cf-v1-*.* in the keras_recommender/training/models.

### Predict Rating

To use the trained CF model to predict the rating of an item by a user, you can use the following code:

```python

from keras_recommender.library.cf import CollaborativeFilteringV1
import pandas as pd

data_dir_path = './data/ml-latest-small' # refers to keras_recommender/training/data/ml-latest-small folder
trained_model_dir_path = './models' # refers to keras_recommender/training/models folder

records = pd.read_csv(data_dir_path + '/ratings.csv')
print(records.describe())

user_id_test = records['userId']
item_id_test = records['movieId']

cf = CollaborativeFilteringV1()
cf.load_model(CollaborativeFilteringV1.get_config_file_path(trained_model_dir_path),
              CollaborativeFilteringV1.get_weight_file_path(trained_model_dir_path))

# batch prediction
predicted_ratings = cf.predict(user_id_test, item_id_test)
print(predicted_ratings)

# individual (user_id, item_id) prediction
for i in range(20):
    user_id = user_id_test[i]
    item_id = item_id_test[i]
    predicted_rating = cf.predict_single(user_id, item_id)
    print('predicted rating: ', predicted_rating)
```


