import numpy as np
import os
from sklearn.cluster import KMeans
from imageHelper import imageHelper
from sklearn.decomposition import PCA, NMF
import pandas as pd

def getKMeansBasedRGBSelectionModel(dir, redimImageSize:tuple, n_clusters=10, pixelsToRemove=[]):
    randomFiles = np.random.choice(os.listdir(dir), 40, replace=False)
    imgRgbArr = []
    for file in randomFiles:
        imgHelper = imageHelper(dir + file)
        imgHelper.imageSize = redimImageSize
        rgbArr = imgHelper.prepImageArray()
        imgRgbArr.extend(rgbArr)

    npRgbArray = np.array(imgRgbArr)

    kMeansModel = KMeans(n_clusters=n_clusters, random_state=42)
    kMeansModel.fit(npRgbArray)

    return kMeansModel

def getRGBDataFrame(model, pixelsToRemove, imgHelper:imageHelper):
    simplifiedDF = getKmeansWithNewRGB(imgHelper.filePath, model, imgHelper.imageSize)
    simplifiedDF = RemoveColors(simplifiedDF, pixelsToRemove)
    return simplifiedDF

def getKmeansWithNewRGB(filePath, model, redimImageSize:list):
    clustersRGB = pd.DataFrame(model.cluster_centers_)
    clustersRGB.columns = ["NewR", "NewG", "NewB"]

    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    cleanedRgbArray =  testRgbArray[np.logical_and(testRgbArray[:, 0] != 0, testRgbArray[:, 1] != 0, testRgbArray[:, 2] != 0), :]
    cleanedRgbArray = cleanedRgbArray[np.logical_and(cleanedRgbArray[:, 0] != 255, cleanedRgbArray[:, 1] != 255, cleanedRgbArray[:, 2] != 255), :]
    testCluster = model.predict(cleanedRgbArray)
    testDf = pd.DataFrame(cleanedRgbArray)
    testDf.columns = ["R", "G", "B"]
    testDf["cluster"] = testCluster

    newDf = testDf.merge(clustersRGB, how="left", right_index=True, left_on="cluster")
    return newDf

def RemoveColors(dataFrame, pixelsToRemove):
    dataFrame.loc[pixelsToRemove, "NewR"] = None
    dataFrame.loc[pixelsToRemove, "NewG"] = None
    dataFrame.loc[pixelsToRemove, "NewB"] = None
    return dataFrame

def ReplaceColors(dataFrame, colorReplacements):
    for cluster, colors in colorReplacements.items():
        dataFrame.loc[dataFrame["cluster"] == cluster, "NewR"] = colors[0]
        dataFrame.loc[dataFrame["cluster"] == cluster, "NewG"] = colors[1]
        dataFrame.loc[dataFrame["cluster"] == cluster, "NewB"] = colors[2]

    return dataFrame

def GetPixelsToReplace(filePath, kMeansModel, redimImageSize:list, clustersToReplace:list):
    dataFrame = getKmeansWithNewRGB(filePath, kMeansModel, redimImageSize)
    return dataFrame['cluster'].isin(clustersToReplace)

def getPCABasedRGBflatteningModel(dir, redimImageSize:tuple):
    randomFiles = np.random.choice(os.listdir(dir), 40, replace=False)
    imgRgbArr = []
    for file in randomFiles:
        imgHelper = imageHelper(dir + file)
        imgHelper.imageSize = redimImageSize
        rgbArr = imgHelper.prepImageArray()
        imgRgbArr.extend(rgbArr)

    npRgbArray = np.array(imgRgbArr)

    pcaModel = PCA(n_components=1)
    pcaModel.fit(npRgbArray)

    return pcaModel

def getPCABasedRGBSelectionModel(dir, redimImageSize:tuple, n_components=50):
    randomFiles = np.random.choice(os.listdir(dir), 40, replace=False)
    imgRgbArr = []
    for file in randomFiles:
        imgHelper = imageHelper(dir + file)
        imgHelper.imageSize = redimImageSize
        rgbArr = imgHelper.prepImageArray()
        imgRgbArr.extend(rgbArr.transpose())

    npRgbArray = np.array(imgRgbArr)

    pcaModel = PCA(n_components=n_components)
    pcaModel.fit(npRgbArray)

    return pcaModel


def getNMFBasedRGBSelectionModel(dir, redimImageSize:tuple, n_components=50, pixelsToRemove=[]):
    randomFiles = np.random.choice(os.listdir(dir), 40, replace=False)
    imgRgbArr = []
    for file in randomFiles:
        imgHelper = imageHelper(dir + file)
        imgHelper.imageSize = redimImageSize
        rgbArr = imgHelper.prepImageArray()
        rgbArr[pixelsToRemove] = [0, 0, 0]
        imgRgbArr.extend(rgbArr.transpose())

    npRgbArray = np.array(imgRgbArr)

    nmfModel = PCA(n_components=n_components)
    nmfModel.fit(npRgbArray)

    return nmfModel