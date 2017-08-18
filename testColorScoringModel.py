from imageHelper import imageHelper
import scoreModelHelper

import numpy as np
import matplotlib.pyplot as plt
from image_score_repository import image_score_repository

def testFlattenAndReshape(filePath, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(testRgbArray))
    plt.imshow(newTestRGB)
    plt.show()


def testSVMBasedColorScoringModel(filePath, scoreModel, redimImageSize:list, pixelsToRemove):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    img_repo = image_score_repository(scoreModel, pixelsToRemove, redimImageSize)
    score = img_repo.get_image_score(imgHelper)
    plt.text(0, 0, 'Score = {0}'.format(score))
    imgHelper.justDisplayImage()
