# keras-recommender
Recommender built using keras

The dataset is taken from  [ml-latest-small (MovieLens)](https://grouplens.org/datasets/movielens/)

The trained models an be found in keras_recommender/training/models

# Deep Learning Models

* Collaborative Filtering V1: hidden factor analysis implementation of CF
    * training: keras_recommender/training/collaborative_filtering_v1.py
* Collaborative Filtering V2: CF with feedforward dense layer
    * training: keras_recommender/training/collaborative_filtering_v2.py
* CNN Content-Based Filtering: Use VGG16 for image simlarity on content-based filtering
    * training: keras_recommender/training/vgg16_content_based_filtering.py


