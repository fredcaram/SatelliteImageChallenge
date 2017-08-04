import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from imageHelper import imageHelper
import rgbSelectionModel

def testFlattenAndReshape(filePath, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(testRgbArray))
    plt.imshow(newTestRGB)
    plt.show()

def colorToFloat(color):
    if(color == 0):
        return 0
    return color / 255

def testKMeansBasedRGBSelectionModel(filePath, model, redimImageSize:list, pixelsToRemove):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    rgbSelectionModel.getRGBDataFrame(model, pixelsToRemove, imgHelper)
    newDf = rgbSelectionModel.getKmeansWithNewRGB(filePath, model, redimImageSize)
    newDf = rgbSelectionModel.RemoveColors(newDf, pixelsToRemove)
    newTestRGB = newDf.loc[:, ["NewR", "NewG", "NewB"]]
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(newTestRGB))
    pointsPerCluster = getKmeansPointPerCluster(filePath, model, redimImageSize)
    colors = [(colorToFloat(r),colorToFloat(g),colorToFloat(b))
              for r,g,b in pointsPerCluster.loc[:,["NewR", "NewG", "NewB"]].values]

    plt.imshow(newTestRGB)
    plt.table(cellText=pointsPerCluster.values, colWidths=[0.2] * len(pointsPerCluster.columns),
              rowLabels=pointsPerCluster.index,
              colLabels=pointsPerCluster.columns,
              cellLoc='center', rowLoc='center',
              rowColours=colors,
              loc='top')
    fig = plt.gcf()

    plt.show()


def getKmeansPointPerCluster(filePath, model, redimImageSize:tuple):
    newDf = rgbSelectionModel.getKmeansWithNewRGB(filePath, model, redimImageSize)
    newDf["points"] = 1
    pointsPerCluster = newDf.groupby(by=["cluster", "NewR", "NewG", "NewB"])["points"].count()
    pointsPerCluster = pointsPerCluster.reset_index()
    pointsPerCluster.reindex(pointsPerCluster["cluster"])
    pointsPerCluster.sort_values("points", ascending=False, inplace=True)
    return pointsPerCluster


def testPCABasedRGBflatteningModel(filePath, model, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    newFeatures = model.transform(testRgbArray)
    reversedFeatures = model.inverse_transform(newFeatures)
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(reversedFeatures))

    plt.imshow(newTestRGB)
    plt.show()

def testPCABasedRGBSelectionModel(filePath, model, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray().transpose()
    newFeatures = model.transform(testRgbArray)
    reversedFeatures = model.inverse_transform(newFeatures)
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(reversedFeatures).transpose())

    plt.imshow(newTestRGB)
    plt.show()

def testNMFBasedRGBSelectionModel(filePath, model, redimImageSize:tuple):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray().transpose()
    newFeatures = model.transform(testRgbArray)
    reversedFeatures = model.inverse_transform(newFeatures)
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(reversedFeatures).transpose())

    plt.imshow(newTestRGB)
    plt.show()