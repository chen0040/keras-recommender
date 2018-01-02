from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import dot, concatenate, Embedding, Input, Flatten, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd

EMBEDDING_SIZE = 100
VERBOSE = 1


class CollaborativeFilteringV1(object):

    model_name = 'cf-v1'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None

    def load_model(self, config_file_path, weight_file_path):
        self.config = np.load(config_file_path).item()
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV1.model_name + '-architecture.json'

    def create_model(self):
        user_id_input = Input(shape=[1], name='user')
        item_id_input = Input(shape=[1], name='item')

        user_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_user_id + 1,
                                   input_length=1, name='user_embedding')(user_id_input)
        item_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_item_id + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        # y = merge([user_vecs, item_vecs], mode='dot', output_shape=(1,))
        y = dot([user_vecs, item_vecs], axes=1)

        model = Model(inputs=[user_id_input, item_id_input], outputs=[y])
        model.compile(optimizer='adam', loss='mae')

        return model

    def fit(self, config, user_id_train, item_id_train, rating_train,  model_dir_path, batch_size=None, epoches=None, validation_split=None):
        if batch_size is None:
            batch_size = 64
        if epoches is None:
            epoches = 20
        if validation_split is None:
            validation_split = 0.1

        self.config = config
        self.max_item_id = config['max_item_id']
        self.max_user_id = config['max_user_id']
        np.save(CollaborativeFilteringV1.get_config_file_path(model_dir_path=model_dir_path), self.config)

        self.model = self.create_model()
        open(CollaborativeFilteringV1.get_architecture_file_path(model_dir_path=model_dir_path), 'w').write(self.model.to_json())

        weight_file_path = CollaborativeFilteringV1.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([user_id_train, item_id_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=VERBOSE, callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        return history

    def predict(self, user_ids, item_ids):
        predicted = self.model.predict([user_ids, item_ids])
        return predicted

    def predict_single(self, user_id, item_id):
        predicted = self.model.predict([pd.Series([user_id]), pd.Series([item_id])])[0][0]
        return predicted

    def evaluate(self, user_id_test, item_id_test, rating_test):
        test_preds = self.model.predict([user_id_test, item_id_test]).squeeze()
        mae = mean_absolute_error(test_preds, rating_test)
        print("Final test MAE: %0.3f" % mae)
        return {'mae': mae}


class CollaborativeFilteringV2(object):
    model_name = 'cf-v2'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None

    def load_model(self, config_file_path, weight_file_path):
        self.config = np.load(config_file_path).item()
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringV2.model_name + '-architecture.json'

    def create_model(self):
        user_id_input = Input(shape=[1], name='user')
        item_id_input = Input(shape=[1], name='item')

        user_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_user_id + 1,
                                   input_length=1, name='user_embedding')(user_id_input)
        item_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_item_id + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        input_vecs = concatenate([user_vecs, item_vecs])
        input_vecs = Dropout(0.5)(input_vecs)

        x = Dense(64, activation='relu')(input_vecs)

        y = Dense(1)(x)

        model = Model(inputs=[user_id_input, item_id_input], outputs=[y])
        model.compile(optimizer='adam', loss='mae')

        return model

    def fit(self, config, user_id_train, item_id_train, rating_train, model_dir_path, batch_size=None, epoches=None, validation_split=None):
        if batch_size is None:
            batch_size = 64
        if epoches is None:
            epoches = 20
        if validation_split is None:
            validation_split = 0.1

        self.config = config
        self.max_item_id = config['max_item_id']
        self.max_user_id = config['max_user_id']
        np.save(CollaborativeFilteringV2.get_config_file_path(model_dir_path=model_dir_path), self.config)

        self.model = self.create_model()
        open(CollaborativeFilteringV2.get_architecture_file_path(model_dir_path=model_dir_path), 'w').write(self.model.to_json())

        weight_file_path = CollaborativeFilteringV2.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([user_id_train, item_id_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=VERBOSE, callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        return history

    def predict(self, user_ids, item_ids):
        predicted = self.model.predict([user_ids, item_ids])
        return predicted

    def evaluate(self, user_id_test, item_id_test, rating_test):
        test_preds = self.model.predict([user_id_test, item_id_test]).squeeze()
        mae = mean_absolute_error(test_preds, rating_test)
        print("Final test MAE: %0.3f" % mae)
        return {'mae': mae}

    def predict_single(self, user_id, item_id):
        predicted = self.model.predict([pd.Series([user_id]), pd.Series([item_id])])[0][0]
        return predicted
    

class CollaborativeFilteringWithTemporalInformation(object):
    model_name = 'temporal-cf'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.max_date = 0
        self.min_date = 0
        self.config = None

    def load_model(self, config_file_path, weight_file_path):
        self.config = np.load(config_file_path).item()
        self.max_user_id = self.config['max_user_id']
        self.max_item_id = self.config['max_item_id']
        self.max_date = self.config['max_date']
        self.min_date = self.config['min_date']
        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + CollaborativeFilteringWithTemporalInformation.model_name + '-architecture.json'

    def create_model(self):
        user_id_input = Input(shape=[1], name='user')
        item_id_input = Input(shape=[1], name='item')
        meta_input = Input(shape=[1], name='meta_item')

        user_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_user_id + 1,
                                   input_length=1, name='user_embedding')(user_id_input)
        item_embedding = Embedding(output_dim=EMBEDDING_SIZE, input_dim=self.max_item_id + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        input_vecs = concatenate([user_vecs, item_vecs, meta_input])
        input_vecs = Dropout(0.5)(input_vecs)

        x = Dense(64, activation='relu')(input_vecs)

        y = Dense(1)(x)

        model = Model(inputs=[user_id_input, item_id_input, meta_input], outputs=[y])
        model.compile(optimizer='adam', loss='mae')

        return model

    def fit(self, config, user_id_train, item_id_train, timestamp_train, rating_train, model_dir_path, batch_size=None, epoches=None, validation_split=None):
        if batch_size is None:
            batch_size = 64
        if epoches is None:
            epoches = 20
        if validation_split is None:
            validation_split = 0.1

        self.config = config
        self.max_item_id = config['max_item_id']
        self.max_user_id = config['max_user_id']

        parsed_dates = [int(str(item_date)[-4:])
                        for item_date in timestamp_train.tolist()]

        parsed_dates = pd.Series(parsed_dates, index=timestamp_train.index)
        max_date = max(parsed_dates)
        min_date = min(parsed_dates)

        self.max_date = max_date
        self.min_date = min_date

        self.config['max_date'] = max_date
        self.config['min_date'] = min_date

        np.save(CollaborativeFilteringWithTemporalInformation.get_config_file_path(model_dir_path=model_dir_path), self.config)

        self.model = self.create_model()
        open(CollaborativeFilteringWithTemporalInformation.get_architecture_file_path(model_dir_path=model_dir_path), 'w').write(self.model.to_json())

        weight_file_path = CollaborativeFilteringWithTemporalInformation.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([user_id_train, item_id_train, timestamp_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=VERBOSE, callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        return history

    def predict(self, user_ids, item_ids, timestamps):
        timestamps = pd.Series([int(str(item_date)[-4:])
                                for item_date in timestamps.tolist()], index=timestamps.index)
        timestamps = (timestamps - self.min_date) / (self.max_date - self.min_date)
        predicted = self.model.predict([user_ids, item_ids, timestamps])
        return predicted

    def evaluate(self, user_id_test, item_id_test, timestamp_test, rating_test):
        timestamp_test = pd.Series([int(str(item_date)[-4:])
                                    for item_date in timestamp_test.tolist()], index=timestamp_test.index)
        timestamp_test = (timestamp_test - self.min_date) / (self.max_date - self.min_date)
        test_preds = self.model.predict([user_id_test, item_id_test, timestamp_test]).squeeze()
        mae = mean_absolute_error(test_preds, rating_test)
        print("Final test MAE: %0.3f" % mae)
        return {'mae': mae}

    def predict_single(self, user_id, item_id, timestamp):
        timestamps = pd.Series([timestamp])
        timestamps = pd.Series([int(str(item_date)[-4:])
                                for item_date in timestamps.tolist()], index=timestamps.index)
        timestamps = (timestamps - self.min_date) / (self.max_date - self.min_date)
        predicted = self.model.predict([pd.Series([user_id]), pd.Series([item_id]), timestamps])[0][0]
        return predicted
