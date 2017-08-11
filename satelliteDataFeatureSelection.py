import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import colorScoringModel
from background_helper import background_helper
from imageHelper import imageHelper

dir = "satelliteImages\\"
kmeansModelFilename = 'kMeansBackgroundSelectionModel.sav'
redimImageSize = [250, 253]
backgroundClusters = 5
rgbsToRemove = [[210, 210, 255],#Water
                [185, 185, 185],#Other Countries
                [255, 255, 255]]#Background

#Remove background, sea and other countries to reduce noise
bgHelper = background_helper(dir, kmeansModelFilename, redimImageSize, backgroundClusters, rgbsToRemove)
pixelsToRemove = bgHelper.get_pixels_to_remove()
#bgHelper.display_image_without_background()

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

fig, axes = plt.subplots(3, sharex=True)
agricultureProduction = pd.read_csv("FAOSTAT\\FAOSTAT_data_8-3-2017.csv")
#Got 3 year before to use for the rolling mean
agricultureProduction = agricultureProduction.loc[agricultureProduction["Year"] >= 1997,]
#Just Net Production
agricultureProduction = agricultureProduction.loc[agricultureProduction["Element Code"] == 154]
agricultureProduction = agricultureProduction.sort_values("Year")
agricultureProductionValue = agricultureProduction.groupby("Year")["Value"].sum()
#The rolling mean was used to represent the growing trend
agricultureProductionRollingMean = agricultureProductionValue.rolling(window=2, center=False).mean()

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
agricultureProductionValueFrom2000 = agricultureProductionValue[agricultureProductionValue.index >= 2000]
agricultureProductionRollingMeanFrom2000 = agricultureProductionRollingMean[agricultureProductionRollingMean.index >= 2000]
agricultureProductionNormalizedFrom2000 = agricultureProductionValueFrom2000 - agricultureProductionRollingMeanFrom2000
prodAndImageCov = np.corrcoef(
    agricultureProductionNormalizedFrom2000[agricultureProductionNormalizedFrom2000.index <= 2013].values,
    imageScoreByYear[imageScoreByYear.index <= 2013].values)

# When comparing the production - rolling mean with the image greenish, a negative correlation was found (~0.44, ~0.42)
#  when using window = 3 and = 2
print(prodAndImageCov)
plt.text(0, 0, 'Correlation = {0}'.format(prodAndImageCov))
agricultureProductionValueFrom2000.plot(ax=axes[0], color='blue')
agricultureProductionRollingMeanFrom2000.plot(ax=axes[0], color='red')
agricultureProductionNormalizedFrom2000.plot(ax=axes[1])
imageScoreByYear.plot(ax=axes[2])
plt.show()

agricultureByYearAndItem = agricultureProduction.groupby(["Year", "Item"])["Value"].mean()
agricultureByYearAndItemRollingMean = agricultureByYearAndItem.rolling(window=3, center=False).mean()

agricultureByYearAndItem = agricultureByYearAndItem.reset_index()
agricultureByYearAndItem2000 = agricultureByYearAndItem[agricultureByYearAndItem["Year"] >= 2000]

agricultureByYearAndItemRollingMean.name = "RollingMean"
agricultureByYearAndItemRollingMean = agricultureByYearAndItemRollingMean.reset_index()
agricultureByYearAndItemRollingMean2000 = agricultureByYearAndItemRollingMean[
    agricultureByYearAndItemRollingMean["Year"] >= 2000]

agricultureByYearAndItem2000 = agricultureByYearAndItem2000.merge(agricultureByYearAndItemRollingMean2000,
                                                                  left_on=["Year", "Item"], right_on=["Year", "Item"])
agricultureByYearAndItem2000["NormalizedValue"] = agricultureByYearAndItem2000["Value"] - agricultureByYearAndItem2000["RollingMean"]

imageScoreByYear = imageScoreByYear.reset_index()
agricultureByYearAndItem2000 = agricultureByYearAndItem2000.merge(imageScoreByYear, left_on="Year", right_on="Year")

#pivotedAgricultureDf = agricultureByYearAndItem2000.pivot(index='Item', columns='Year', values=['RollingMean', 'Score'])
#print(meltedAgricultureDf.head())

itemCorrelations = []
items = np.unique(agricultureByYearAndItem2000["Item"].values)
for item in items:
    agriculturePerItem = agricultureByYearAndItem2000[agricultureByYearAndItem2000["Item"] == item]
    corr =  np.corrcoef(
        agriculturePerItem["NormalizedValue"].values,
        agriculturePerItem["Score"].values)
    itemCorrelations.append(corr[0,1])

aggriculturePerItemCorrelation = pd.DataFrame({"Item": items, "Correlation": itemCorrelations})
#print(aggriculturePerItemCorrelation)
#g = sns.FacetGrid(meltedAgricultureDf, hue="Item", row='variable', sharex=True, sharey=False)
#g = g.map(sns.pointplot, "Year", "value")
#plt.show()

top10corrAsc = aggriculturePerItemCorrelation.sort_values("Correlation").head(10)
top10corrDesc = aggriculturePerItemCorrelation.sort_values("Correlation", ascending=False).head(10)
print(top10corrAsc)
print(top10corrDesc)

meltedAgricultureDf = pd.melt(agricultureByYearAndItem2000, id_vars=['Item', 'Year'], value_vars=['Score', 'NormalizedValue'])
top10AscCorrAgricultureDf = meltedAgricultureDf.merge(top10corrAsc, how="inner", left_on="Item", right_on="Item")
top10DescCorrAgricultureDf = meltedAgricultureDf.merge(top10corrDesc, how="inner", left_on="Item", right_on="Item")

#Top 10 correlations
g1 = sns.FacetGrid(top10AscCorrAgricultureDf, col="variable", row='Item', sharex=True, sharey=False, size=7)
g1 = g1.map(sns.pointplot, "Year", "value")
plt.show()

g2 = sns.FacetGrid(top10DescCorrAgricultureDf, col="variable", row='Item', sharex=True, sharey=False, size=7)
g2 = g2.map(sns.pointplot, "Year", "value")
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