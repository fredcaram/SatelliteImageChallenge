from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

class imageHelper:
    def __init__(self, filePath: str):
        self.filePath = filePath
        self.imageSize = []

    def imageExists(self):
        return os.path.isfile(self.filePath)


    def getResizedImage(self):
        img = misc.imread(self.filePath, mode='RGBA')
        if (len(self.imageSize) == 0):
            self.imageSize = [img.shape[0], img.shape[1]]
        resizedImg = misc.imresize(img, self.imageSize)
        return resizedImg

    def prepImageArray(self):
        resizedImg = self.getResizedImage()
        rgbArr = np.delete(resizedImg, 3, axis=2)
        rgbArr = self.rgbTo2Dimensional(rgbArr)
        return rgbArr


    def rgbTo2Dimensional(self, threeDimRGBArray: np.array):
        rgbArr = np.reshape(threeDimRGBArray, (-1, 3))
        return rgbArr


    def rgbTo3Dimensional(self, twoDimRGBArray: np.array):
        flattenImgArr = twoDimRGBArray.flatten()
        threeDSize = list(self.imageSize)
        threeDSize.append(3)
        rgbArr = np.reshape(flattenImgArr.transpose(), threeDSize)
        return rgbArr

    def justDisplayImage(self):
        testRgbArray = self.getResizedImage()
        plt.imshow(testRgbArray)
        plt.show()