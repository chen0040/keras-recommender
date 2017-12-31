from keras.applications import VGG16
import numpy as np
import os


class Vgg16ContentBaseFiltering(object):

    def __init__(self):
        self.matrix_res = None
        self.similarity_deep = None
        self.model = VGG16(include_top=False, weights='imagenet')
        self.matrix_idx_to_item_id = None
        self.item_id_to_matrix_idx = None

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
        if not os.path.exists(predictions_file):
            matrix_res = np.zeros([num_items, 25088])
            for i, (item_id, item_img) in enumerate(item_id2img.items()):
                print('Predicting for item = {}'.format(item_id))
                matrix_res[i, :] = self.model.predict(item_img).ravel()
            np.savez_compressed(file=predictions_file, matrix_res=matrix_res)
        else:
            matrix_res = np.load(predictions_file)['matrix_res']

        similarity_deep = matrix_res.dot(matrix_res.T)
        norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
        similarity_deep = similarity_deep / (norms * norms.T)
        print(similarity_deep.shape)
        self.similarity_deep = similarity_deep

    def recommend_closest_items(self, target_item_id, top_k=None):
        if top_k is None:
            top_k = 20
        target_idx = self.item_id_to_matrix_idx[target_item_id]
        closest_matrix_idx_list = np.argsort(self.similarity_deep[target_idx, :])[::-1][0:top_k]
        closest_items_ids = [self.matrix_idx_to_item_id[matrix_id] for matrix_id in closest_matrix_idx_list]
        return closest_items_ids
