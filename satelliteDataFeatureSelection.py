import pickle
import numpy as np
import os
import pandas as pd
import testColorScoringModel
from imageHelper import imageHelper
import colorScoringModel
import matplotlib.pyplot as plt
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
rgbsToRemove = [[210, 210, 255],#Water
                [185, 185, 185],#Other Countries
                [255, 255, 255]]#Background
clustersToRemove = kMeansmodel.predict(rgbsToRemove)
#clustersToReplace = [0, 2, 4]
pixelsToRemove = rgbSelectionModel.GetPixelsToReplace(dir + testFile, kMeansmodel, redimImageSize, clustersToRemove)
#testRGBSelectionModel.testKMeansBasedRGBSelectionModel(dir + testFile, kMeansmodel, redimImageSize, pixelsToRemove)

#Use kmeans to select color features for model training
#With k-mean = 10 I got only a 0.2672 F-measure score for the score model
#nclusters = 16
#kmeansRGBModelFilename = 'kMeansRGBSelectionModel.sav'
#kmeansRGBModel =rgbSelectionModel.getKMeansBasedRGBSelectionModel(dir, redimImageSize, nclusters, pixelsToRemove)
#pickle.dump(kmeansRGBModel, open(kmeansRGBModelFilename, 'wb'))
#kmeansRGBModel = pickle.load(open(kmeansRGBModelFilename, 'rb'))
#testRGBSelectionModel.testKMeansBasedRGBSelectionModel(dir + testFile, kmeansRGBModel, redimImageSize, pixelsToRemove)

#ncomponents = 60
#nmfRGBModelFilename = 'nmfRGBSelectionModel'
#nmfRGBModel = rgbSelectionModel.getNMFBasedRGBflatteningModel(dir, redimImageSize, pixelsToRemove)
#pickle.dump(nmfRGBModel, open(nmfRGBModelFilename, 'wb'))
#nmfRGBModel = pickle.load(open(nmfRGBModelFilename, 'rb'))
#testRGBSelectionModel.testNMFBasedRGBflatteningModel(dir + testFile, nmfRGBModel, redimImageSize)

#Plot the original image
#testRGBSelectionModel.testFlattenAndReshape(dir + testFile, redimImageSize)


#categorize between yellow and green (other colors are discarded)
colorScaleDir = "colorScales\\"
colorScaleDF = pd.DataFrame()
redimColorScaleSize = [100, 50]
#The score increase or decrease according ot the color intensity (Must be integer)
colorIntensityWeight = 1
#Increase the score gap between two different colors, so that the difference between green and yellow increases
colorGap = 0

numberOfScaleFiles = 3
for i in range(1,numberOfScaleFiles + 1):
    greenScore =  colorGap + colorIntensityWeight * (numberOfScaleFiles + i)
    yellowScore = colorIntensityWeight * (numberOfScaleFiles - i)
    greenRgbDf = colorScoringModel.getImageDfWithScore(colorScaleDir + "green{0}.png".format(i),
                                                     greenScore, redimColorScaleSize)
    colorScaleDF = colorScaleDF.append(greenRgbDf)

    yellowRgbDf = colorScoringModel.getImageDfWithScore(colorScaleDir + "yellow{0}.png".format(i),
                                                        yellowScore, redimColorScaleSize)
    colorScaleDF = colorScaleDF.append(yellowRgbDf)

otherColorsDf = colorScoringModel.getImageDfWithScore(colorScaleDir + "otherColors.png",
                                                      0, redimColorScaleSize)
colorScaleDF = colorScaleDF.append(otherColorsDf)

svmScoringModelFile = 'svmScoringModel.sav'
#0.98 f-measure
#scoreModel = colorScoringModel.trainScoreClassificationModel(colorScaleDF, True)
#pickle.dump(scoreModel, open(svmScoringModelFile, 'wb'))
#scoreModel = pickle.load(open(svmScoringModelFile, 'rb'))

#0.99 f-measure
rfScoringModelFile = 'rfScoringModel.sav'
#scoreModel = colorScoringModel.trainRFScoreClassificationModel(colorScaleDF, True)
#pickle.dump(scoreModel, open(rfScoringModelFile, 'wb'))
#scoreModel = pickle.load(open(rfScoringModelFile, 'rb'))

#0.99 f-measure
etScoringModelFile = 'etScoringModel.sav'
#scoreModel = colorScoringModel.trainETScoreClassificationModel(colorScaleDF, True)
#pickle.dump(scoreModel, open(etScoringModelFile, 'wb'))
scoreModel = pickle.load(open(etScoringModelFile, 'rb'))

#Plot one yellow and one green image with the score for comparing
yellishImage = "map_BRA_y2000_w36.png"
greenishImage = "map_BRA_y2005_w16.png"
#testColorScoringModel.testSVMBasedColorScoringModel(dir + yellishImage, scoreModel, redimImageSize, pixelsToRemove)
#testColorScoringModel.testSVMBasedColorScoringModel(dir + greenishImage, scoreModel, redimImageSize, pixelsToRemove)

#Plot ten random images with the score for intuition
#testingFiles = np.random.choice(os.listdir(dir), 10, replace=False)
#for testingFile in testingFiles:
#    testColorScoringModel.testSVMBasedColorScoringModel(dir + testingFile, scoreModel, redimImageSize, pixelsToRemove)

fig, axes = plt.subplots(2)
agricultureProduction = pd.read_csv("FAOSTAT\\FAOSTAT_data_8-3-2017.csv")
#Got 3 year before to use for the rolling mean
agricultureProduction = agricultureProduction.loc[agricultureProduction["Year"] >= 1997,]
agricultureProductionValue = agricultureProduction.sort_values("Year").groupby("Year")["Value"].mean()
agricultureProductionRollingMean = agricultureProductionValue.rolling(window=3,center=True).mean()

ind = 0
imageScores = []
country = "BRA"
yearRange = agricultureProduction["Year"].unique()
for year in yearRange:
    for week in range(1, 52):
        imagePath = 'satelliteImages\map_{0}_y{1}_w{2}.png'.format(country, year, week)
        imgHelper = imageHelper(imagePath)
        if imgHelper.imageExists():
            imgHelper.imageSize = redimImageSize
            score = colorScoringModel.GetImageScore(imgHelper, pixelsToRemove, scoreModel)
            imageScores.append([country, year, week, score])
            ind += 1

imageScoresDf = pd.DataFrame(imageScores)
imageScoresDf.columns = ["Country", "Year", "Week", "Score"]
imageScoreByYear = imageScoresDf.groupby("Year")["Score"].mean()

#Plot productionXscore plot
agricultureProductionValue[agricultureProductionValue.index >= 2000,:].plot(ax=axes[0], color='blue')
agricultureProductionRollingMean[agricultureProductionRollingMean.index >= 2000,:].plot(ax=axes[0], color='red')
imageScoreByYear.plot(ax=axes[1])

plt.show()

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