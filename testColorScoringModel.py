from imageHelper import imageHelper
import rgbSelectionModel

import numpy as np
import matplotlib.pyplot as plt

def testFlattenAndReshape(filePath, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(testRgbArray))
    plt.imshow(newTestRGB)
    plt.show()


def testSVMBasedColorScoringModel(filePath, rgbModel, scoreModel, redimImageSize:list, pixelsToRemove):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    newDf = rgbSelectionModel.getRGBDataFrame(rgbModel, pixelsToRemove, imgHelper)
    newTestRGB = newDf.loc[:, ["NewR", "NewG", "NewB"]]
    #newTestRGB = imgHelper.rgbTo3Dimensional(np.array(newTestRGB))
    scoreArr = scoreModel.predict(np.array(newTestRGB.values))

    plt.imshow(newTestRGB)
    plt.table(cellText=pointsPerCluster.values, colWidths=[0.2] * len(pointsPerCluster.columns),
              rowLabels=pointsPerCluster.index,
              colLabels=pointsPerCluster.columns,
              cellLoc='center', rowLoc='center',
              rowColours=colors,
              loc='top')
    fig = plt.gcf()