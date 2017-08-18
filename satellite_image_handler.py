import numpy as np
import os

import testColorScoringModel
import testRGBSelectionModel
from background_helper import background_helper
from color_scoring_model import color_scoring


class SatelliteImageHandler:
    def __init__(self):
        self.satellite_image_dir = "satelliteImages\\"
        self.satellite_image_size = [250, 253]
        self.background_model_file_name = 'kMeansBackgroundSelectionModel.sav'
        self.number_of_bg_clusters = 5
        self.rgbs_to_remove = [[210, 210, 255],#Water
                        [185, 185, 185],#Other Countries
                        [255, 255, 255]]#Background
        self.background_helper = background_helper(self.satellite_image_dir,
                                     self.background_model_file_name,
                                     self.satellite_image_size, self.number_of_bg_clusters, self.rgbs_to_remove)

    def get_pixels_to_remove(self):
        return self.background_helper.get_pixels_to_remove()

    def visualize_original_image(self, file):
        # Plot the original image
        testRGBSelectionModel.testFlattenAndReshape(self.satellite_image_dir + file, self.satellite_image_size)

    def visualize_background_removal(self):
        self.background_helper.display_image_without_background()

    def build_image_model(self):
        #Remove background, sea and other countries to reduce noise
        pixelsToRemove = self.get_pixels_to_remove()

        #With k-mean from 5 to 20 maximum of 0.2672 F-measure score for the score model
        #With nmf for model flattening 0.60 F-measure score for the score model


        #categorize between yellow and green (other colors are discarded)
        colorScaleDir = "colorScales\\"
        redimColorScaleSize = [100, 50]

        colorGap = 0

        clr_scoring = color_scoring(colorScaleDir)
        clr_scoring.image_size = redimColorScaleSize
        clr_scoring.print_model_score = True

        #0.98 f-measure for svm model

        #0.99 f-measure for random forest model

        #0.99999 f-measure
        et_scoring_model_file = 'etScoringModel.sav'
        scoreModel = clr_scoring.get_model(et_scoring_model_file)
        return scoreModel

    def visualize_model_with_opposite_images(self, scoreModel):
        #Plot one yellow and one green image with the score for comparing
        yellishImage = "map_BRA_y2000_w36.png"
        greenishImage = "map_BRA_y2005_w16.png"
        pixels_to_remove = self.get_pixels_to_remove()
        testColorScoringModel.testSVMBasedColorScoringModel(self.satellite_image_dir + yellishImage,
                                                            scoreModel, self.satellite_image_size, pixels_to_remove)
        testColorScoringModel.testSVMBasedColorScoringModel(self.satellite_image_dir + greenishImage,
                                                            scoreModel, self.satellite_image_size, pixels_to_remove)


    def visualize_model_with_random_images(self, scoreModel, numberOfImages = 10):
        #Plot ten random images with the score for intuition
        pixels_to_remove = self.get_pixels_to_remove()
        testingFiles = np.random.choice(os.listdir(self.satellite_image_dir), numberOfImages, replace=False)
        for testingFile in testingFiles:
            testColorScoringModel.testSVMBasedColorScoringModel(self.satellite_image_dir + testingFile,
                                                                scoreModel, self.satellite_image_size, pixels_to_remove)