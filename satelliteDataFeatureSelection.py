import pickle
import numpy as np
import os
import pandas as pd
from imageHelper import imageHelper
import colorScoringModel
from sklearn.model_selection import train_test_split

import rgbSelectionModel
import testRGBSelectionModel

dir = "satelliteImages\\"
kmeansModelFilename = 'kMeansBackgroundSelectionModel.sav'
redimImageSize = [250, 253]

#Remove background, sea and other countries to reduce noise
backgroundClusters = 5
#kMeansmodel = rgbSelectionModel.getKMeansBasedRGBSelectionModel(dir, redimImageSize, backgroundClusters)
#pickle.dump(kMeansmodel, open(kmeansModelFilename, 'wb'))
kMeansmodel = pickle.load(open(kmeansModelFilename, 'rb'))

testFile = np.random.choice(os.listdir(dir), replace=False,)
#clusterPoints = testRGBSelectionModel.getKmeansPointPerCluster(dir + testFile, kMeansmodel, redimImageSize)
clustersToReplace = [0, 2, 4]
pixelsToRemove = rgbSelectionModel.GetPixelsToReplace(dir + testFile, kMeansmodel, redimImageSize, clustersToReplace)
#testRGBSelectionModel.testKMeansBasedRGBSelectionModel(dir + testFile, kMeansmodel, redimImageSize, pixelsToRemove)

#Use kmeans to select color features for model training
nclusters = 10
kmeansRGBModelFilename = 'kMeansRGBSelectionModel.sav'
#kmeansRGBModel =rgbSelectionModel.getKMeansBasedRGBSelectionModel(dir, redimImageSize, nclusters, pixelsToRemove)
#pickle.dump(kmeansRGBModel, open(kmeansRGBModelFilename, 'wb'))
kmeansRGBModel = pickle.load(open(kmeansRGBModelFilename, 'rb'))
#testRGBSelectionModel.testKMeansBasedRGBSelectionModel(dir + testFile, kmeansRGBModel, redimImageSize, pixelsToRemove)

#Plot the original image
#testRGBSelectionModel.testFlattenAndReshape(dir + testFile, redimImageSize)


#yellow will be categorized from 1 to 5 according to the intensity, green will be from 6 to 10
colorScaleDir = "colorScales\\"
colorScaleDF = pd.DataFrame()
redimColorScaleSize = [100, 50]
for i in range(1,5):
    greenRgbDf = colorScoringModel.getDfWithScore(colorScaleDir + "green{0}.png".format(i), kmeansRGBModel, i + 5, redimColorScaleSize)
    colorScaleDF = colorScaleDF.append(greenRgbDf)

    yellowRgbDf = colorScoringModel.getDfWithScore(colorScaleDir + "green{0}.png".format(i), kmeansRGBModel, i, redimColorScaleSize)
    colorScaleDF = colorScaleDF.append(yellowRgbDf)

svmScoringModelFile = 'svmScoringModel.sav'
scoreModel = colorScoringModel.trainScoreClassificationModel(colorScaleDF, True)
pickle.dump(scoreModel, open(svmScoringModelFile, 'wb'))
#scoreModel = pickle.load(open(svmScoringModelFile, 'rb'))

yellishImage = "map_BRA_y2000_w36.png"
greenishImage = "map_BRA_y2005_w16.png"

#nmfModelFilename = 'kMeansBackgroundSelectionModel.sav'
#nmfModel = rgbSelectionModel.getNMFBasedRGBSelectionModel(dir, redimImageSize, ncomponents, pixelsToRemove)
#pickle.dump(nmfModel, open(nmfModelFilename, 'wb'))
#nmfModel = pickle.load(open(nmfModelFilename, 'rb'))
#testRGBSelectionModel.testNMFBasedRGBSelectionModel(dir + testFile, nmfModel, redimImageSize)

#pcaModelFilename = 'pcaSelectionModel.sav'

#pcaModel = rgbSelectionModel.getPCABasedRGBSelectionModel(dir, redimImageSize, ncomponents)
#model = rgbSelectionModel.getPCABasedRGBflatteningModel(dir, redimImageSize)

#pickle.dump(kMeansmodel, open(kmeansModelFilename, 'wb'))
#pickle.dump(pcaModel, open(pcaModelFilename, 'wb'))
#pickle.dump(nmfModel, open(nmfModelFilename, 'wb'))


#pcaModel = pickle.load(open(pcaModelFilename, 'rb'))

#testRGBSelectionModel.testPCABasedRGBSelectionModel(dir + testFile, pcaModel, redimImageSize)
#testRGBSelectionModel.testNMFBasedRGBSelectionModel(dir + testFile, nmfModel, redimImageSize)
#testRGBSelectionModel.testPCABasedRGBflatteningModel(dir + testFile, model, redimImageSize)