from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import dot, concatenate, Embedding, Input, Flatten, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_absolute_error

EMBEDDING_SIZE = 100


class CollaborativeFilteringV1(object):

    model_name = 'cf-v1'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None

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
        open(CollaborativeFilteringV1.get_architecture_file_path(model_dir_path=model_dir_path)).write(self.model.to_json())

        weight_file_path = CollaborativeFilteringV1.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([user_id_train, item_id_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=2, callbacks=[checkpoint])
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


class CollaborativeFilteringV2(object):
    model_name = 'cf-v2'

    def __init__(self):
        self.model = None
        self.max_user_id = 0
        self.max_item_id = 0
        self.config = None

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
        open(CollaborativeFilteringV2.get_architecture_file_path(model_dir_path=model_dir_path)).write(self.model.to_json())

        weight_file_path = CollaborativeFilteringV2.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit([user_id_train, item_id_train], rating_train,
                                 batch_size=batch_size, epochs=epoches, validation_split=validation_split,
                                 shuffle=True, verbose=2, callbacks=[checkpoint])
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
