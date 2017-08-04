from imageHelper import imageHelper
from sklearn import svm
import rgbSelectionModel
import numpy as np
from sklearn.model_selection import train_test_split


def getDfWithScore(filePath, model, score, redimImageSize):
    imgHelper = imageHelper(filePath)
    imgHelper.imageSize = redimImageSize
    df = rgbSelectionModel.getKmeansWithNewRGB(imgHelper.filePath, model, imgHelper.imageSize)
    df["score"] = score
    return df

def trainScoreClassificationModel(scoreDf, printModelScore:bool):
    scoreDf["cluster"] = scoreDf["cluster"].astype('category')
    X = np.array(scoreDf.loc[:, ["cluster"]].values)
    y = np.array(scoreDf.loc[:, ["score"]].values)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    if printModelScore:
        print(clf.score(X_test, y_test))

    return clf