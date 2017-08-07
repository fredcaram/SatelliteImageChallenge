from imageHelper import imageHelper
import colorScoringModel

import numpy as np
import matplotlib.pyplot as plt

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
    score = colorScoringModel.GetImageScore(imgHelper, pixelsToRemove, scoreModel)
    plt.text(0, 0, 'Score = {0}'.format(score))
    imgHelper.justDisplayImage()
