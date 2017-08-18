from imageHelper import imageHelper
from sklearn import svm
import rgbSelectionModel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def getKMeansDfWithScore(filePath, model, score, redimImageSize):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    df = rgbSelectionModel.getKmeansWithNewRGB(imgHelper.filePath, model, imgHelper.imageSize)
    df["score"] = score
    df["feature"] = df["cluster"]
    return df

def getNMFDfWithScore(filePath, model, score, redimImageSize):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray().transpose()
    newFeatures = model.transform(testRgbArray)
    reversedFeatures = model.inverse_transform(newFeatures)
    newTestRGB = imgHelper.rgbTo3Dimensional(np.array(reversedFeatures).transpose())
    df = pd.DataFrame(newTestRGB)
    df.columns = ["NewR", "NewG", "NewB"]
    #df["feature"] = newFeatures
    df["score"] = score
    return df

def getFlattenedNMFDfWithScore(filePath, model, score, redimImageSize):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    testRgbArray = imgHelper.prepImageArray()
    newFeatures = model.transform(testRgbArray)
    #reversedFeatures = model.inverse_transform(newFeatures)
    #newTestRGB = imgHelper.rgbTo3Dimensional(np.array(reversedFeatures))
    df = pd.DataFrame(testRgbArray)
    df.columns = ["NewR", "NewG", "NewB"]
    df["feature"] = newFeatures
    df["score"] = score
    return df

def trainFlattenedNMFScoreClassificationModel(scoreDf, printModelScore:bool):
    #scoreDf["feature"] = scoreDf["feature"].astype('category')
    X = np.array(scoreDf.loc[:, ["feature"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf

def trainKMeansScoreClassificationModel(scoreDf, printModelScore:bool):
    #scoreDf["feature"] = scoreDf["feature"].astype('category')
    X = np.array(scoreDf.loc[:, ["cluster"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf

def trainNMFScoreClassificationModel(scoreDf, printModelScore:bool):
    #scoreDf["feature"] = scoreDf["feature"].astype('category')
    X = np.array(scoreDf.loc[:, ["NewR", "NewG", "NewB"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf


def trainScoreClassificationModel(scoreDf, printModelScore:bool):
    X = np.array(scoreDf.loc[:, ["NewR", "NewG", "NewB"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf


def trainRFScoreClassificationModel(scoreDf, printModelScore:bool):
    X = np.array(scoreDf.loc[:, ["NewR", "NewG", "NewB"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf