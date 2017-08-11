import os
import pickle

import numpy as np

import rgbSelectionModel
import testRGBSelectionModel

from imageHelper import imageHelper

class background_helper:
    def __init__(self, dir: str, modelfilename: str, imagesize: list, n_clusters: int, rgbs_to_remove: list):
        self.dir = dir
        self.modelfilename = modelfilename
        self.imagesize = imagesize
        self.n_clusters = n_clusters
        self.rgbs_to_remove = rgbs_to_remove

    def __get_model__(self):
        imgHelper = imageHelper(self.modelfilename)
        if imgHelper.imageExists():
            kmeans_model = pickle.load(open(self.modelfilename, 'rb'))
        else:
            kmeans_model = rgbSelectionModel.getKMeansBasedRGBSelectionModel(self.dir, self.imagesize, self.n_clusters)
            pickle.dump(kmeans_model, open(self.modelfilename, 'wb'))

        return kmeans_model

    def __get_clusters_to_remove__(self):
        kmeans_model = self.__get_model__()
        clustersToRemove = kmeans_model.predict(self.rgbs_to_remove)
        return clustersToRemove

    def get_pixels_to_remove(self):
        kmeans_model = self.__get_model__()
        testFile = np.random.choice(os.listdir(self.dir), replace=False, )
        clusters_to_remove = self.__get_clusters_to_remove__()
        pixels_to_remove = rgbSelectionModel.GetPixelsToReplace(self.dir + testFile, kmeans_model, self.imagesize,
                                                                clusters_to_remove)
        return pixels_to_remove

    def display_image_without_background(self):
        kmeans_model = self.__get_model__()
        testFile = np.random.choice(os.listdir(self.dir), replace=False, )
        pixels_to_remove = self.get_pixels_to_remove()
        testRGBSelectionModel.testKMeansBasedRGBSelectionModel(self.dir + testFile, kmeans_model, self.imagesize,
                                                               pixels_to_remove)