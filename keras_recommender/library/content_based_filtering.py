from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import dot, concatenate, Embedding, Input, Flatten, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
from keras.applications import VGG16

EMBEDDING_SIZE = 100
VERBOSE = 1


class Vgg16ContentBaseFiltering(object):

    model_name = 'vgg16-cbf'

    def __init__(self):
        self.matrix_res = None
        self.similarity_deep = None
        self.model = VGG16(include_top=False, weights='imagenet')
        self.matrix_idx_to_item_id = None
        self.item_id_to_matrix_idx = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        predictions_file = model_dir_path + '/' + Vgg16ContentBaseFiltering.model_name + '-matrix_res.npz'
        return predictions_file

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Vgg16ContentBaseFiltering.model_name + '-similarity-matrix.npy'

    def load_model(self, config_file_path, weight_file_path):
        data = np.load(Vgg16ContentBaseFiltering.get_weight_file_path(model_dir_path=config_file_path))
        self.matrix_res = data['matrix_res']
        self.item_id_to_matrix_idx = data['id2idx']
        self.matrix_idx_to_item_id = data['idx2id']

        self.similarity_deep = np.load(weight_file_path)['similarity_deep']

    def fit(self, item_id2img, model_dir_path=None):
        if model_dir_path is None:
            model_dir_path = '../train/models'

        self.matrix_idx_to_item_id = dict()
        self.item_id_to_matrix_idx = dict()
        for i, (imdb_id, img) in enumerate(item_id2img.items()):  # imdb ids.
            self.matrix_idx_to_item_id[i] = imdb_id
            self.item_id_to_matrix_idx[imdb_id] = i

        num_items = len(item_id2img.keys())
        print('Number of items = {}'.format(num_items))
        predictions_file = model_dir_path + '/matrix_res.npz'

        self.matrix_res = np.zeros([num_items, 25088])
        for i, (item_id, item_img) in enumerate(item_id2img.items()):
            print('Predicting for item = {}'.format(item_id))
            self.matrix_res[i, :] = self.model.predict(item_img).ravel()
        np.savez_compressed(file=predictions_file, matrix_res=self.matrix_res, idx2id=self.matrix_idx_to_item_id, id2idx=self.item_id_to_matrix_idx)

        similarity_deep = self.matrix_res.dot(self.matrix_res.T)
        norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
        similarity_deep = similarity_deep / (norms * norms.T)
        print(similarity_deep.shape)
        self.similarity_deep = similarity_deep
        np.savez_compressed(file=Vgg16ContentBaseFiltering.get_weight_file_path(model_dir_path),
                            similarity_deep=self.similarity_deep)

    def recommend_closest_items(self, target_item_id, top_k=None):
        if top_k is None:
            top_k = 20
        target_idx = self.item_id_to_matrix_idx[target_item_id]
        closest_matrix_idx_list = np.argsort(self.similarity_deep[target_idx, :])[::-1][0:top_k]
        closest_items_ids = [self.matrix_idx_to_item_id[matrix_id] for matrix_id in closest_matrix_idx_list]
        return closest_items_ids


class TemporalContentBasedFiltering(object):
    model_name = 'temporal-cbf'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.max_date = 0
        self.min_date = 0
        self.config = None

    def load_model(self, config_file_path, weight_file_path):
        self.config = np.load(config_file_path).item()
        self.max_item_id = self.config['max_item_id']
        self.max_date = self.config['max_date']
        self.min_date = self.config['min_date']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + TemporalContentBasedFiltering.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + TemporalContentBasedFiltering.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + TemporalContentBasedFiltering.model_name + '-architecture.json'

    def create_model(self):

        # For each sample we input the integer identifiers
        # of a single user and a single item
        item_id_input = Input(shape=[1], name='item')
        meta_input = Input(shape=[1], name='meta_item')

        embedding_size = 32
        item_embedding = Embedding(output_dim=embedding_size, input_dim=self.max_item_id + 1,
                                   input_length=1, name='item_embedding')(item_id_input)
        item_vecs = Flatten()(item_embedding)

        input_vecs = concatenate([item_vecs, meta_input])

        x = Dense(64, activation='relu')(input_vecs)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(input_vecs)
        y = Dense(1)(x)

        model = Model(inputs=[item_id_input, meta_input], outputs=y)
        model.compile(optimizer='adam', loss='mae')

        return model

    def fit(self, config, item_id_train, date_train, rating_train, model_dir_path, batch_size=None, epoches=None, validation_split=None):
        if batch_size is None:
            batch_size = 64
        if epoches is None:
            epoches = 20
        if validation_split is None:
            validation_split = 0.1

        self.config = config
        self.max_item_id = config['max_item_id']

        parsed_dates = [int(str(item_date)[-4:])
                        for item_date in date_train.tolist()]

        parsed_dates = pd.Series(parsed_dates, index=date_train.index)
        max_date = max(parsed_dates)
        min_date = min(parsed_dates)

        self.max_date = max_date
        self.min_date = min_date

        self.config['max_date'] = max_date
        self.config['min_date'] = min_date

        date_train = (parsed_dates - min_date) / (max_date - min_date)

        np.save(TemporalContentBasedFiltering.get_config_file_path(model_dir_path=model_dir_path), self.config)

        self.model = self.create_model()
        open(TemporalContentBasedFiltering.get_architecture_file_path(model_dir_path=model_dir_path), 'w').write(self.model.to_json())

        weight_file_path = TemporalContentBasedFiltering.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([item_id_train, date_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=VERBOSE, callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        return history

    def predict(self, item_ids, dates):
        dates = (dates - self.min_date) / (self.max_date - self.min_date)
        predicted = self.model.predict([item_ids, dates])
        return predicted

    def evaluate(self, item_id_test, dates_test, rating_test):
        dates_test = (dates_test - self.min_date) / (self.max_date - self.min_date)
        test_preds = self.model.predict([item_id_test, dates_test]).squeeze()
        mae = mean_absolute_error(test_preds, rating_test)
        print("Final test MAE: %0.3f" % mae)
        return {'mae': mae}

    def predict_single(self, item_id, date):
        predicted = self.model.predict([pd.Series([item_id]), pd.Series([date])])[0][0]
        return predicted