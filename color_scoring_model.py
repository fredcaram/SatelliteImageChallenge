import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from imageHelper import imageHelper


class color_scoring:
    def __init__(self, dir):
        self.dir = dir
        self.image_size = []
        # Increase the score gap between two different colors, so that the difference between green and yellow increases
        self.color_gap = 0
        # The score increase or decrease according ot the color intensity (Must be integer)
        self.color_intensity_weight = 1
        self.print_model_score = False

    def __get_image_with_score__(self, filePath, score):
        imgHelper = imageHelper(filePath)
        imgHelper.imageSize = self.image_size
        rgbArray = imgHelper.prepImageArray()
        # newTestRGB = imgHelper.rgbTo3Dimensional(np.array(rgbArray))
        df = pd.DataFrame(rgbArray)
        df.columns = ["NewR", "NewG", "NewB"]
        df["score"] = score
        return df

    def get_model(self, model_filename):
        img_helper = imageHelper(model_filename)
        if img_helper.imageExists():
            kmeans_model = pickle.load(open(model_filename, 'rb'))
        else:
            kmeans_model = ()
            pickle.dump(kmeans_model, open(model_filename, 'wb'))

        return kmeans_model

    def __build_color_scale_df__(self):
        color_scale_df = pd.DataFrame()
        n_scale_files = 3
        for i in range(1, n_scale_files + 1):
            green_score = self.color_gap + self.color_intensity_weight * (n_scale_files + i)
            yellow_score = self.color_intensity_weight * (n_scale_files - i)
            green_rgb_df = self.__get_image_with_score__(self.dir + "green{0}.png".format(i),
                                                              green_score)
            color_scale_df = color_scale_df.append(green_rgb_df)

            yellow_rgb_df = self.__get_image_with_score__(self.dir + "yellow{0}.png".format(i),
                                                               yellow_score)
            color_scale_df = color_scale_df.append(yellow_rgb_df)
        other_colors_df = self.__get_image_with_score__(self.dir + "otherColors.png",
                                                             0)
        color_scale_df = color_scale_df.append(other_colors_df)
        return color_scale_df

    def trains_et_score_classification_model(self):
        score_df = self.__build_color_scale_df__()
        X = np.array(score_df.loc[:, ["NewR", "NewG", "NewB"]].values)
        y = np.array(score_df.loc[:, ["score"]].values)
        y = np.ravel(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ExtraTreesClassifier(n_estimators=20)
        clf.fit(X_train, y_train)

        if self.print_model_score:
            print(clf.score(X_test, y_test))

        return clf