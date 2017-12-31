from keras.applications import VGG16
import numpy as np
import os


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
