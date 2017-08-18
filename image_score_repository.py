import numpy as np
import pandas as pd

from imageHelper import imageHelper


class image_score_repository:
    def __init__(self, score_model, pixels_to_remove: list, satellite_image_size: list,
                 image_score_filename = "image_score.csv"):
        self.country = "BRA"
        self.satellite_dir = 'satelliteImages\\'
        self.score_model = score_model
        self.pixels_to_remove = pixels_to_remove
        self.satellite_image_size = satellite_image_size
        self.image_score_filename = image_score_filename

    def get_image_score(self, img_helper):
        rgb_array = img_helper.prepImageArray()
        rgb_array = rgb_array[np.logical_not(self.pixels_to_remove)]
        prediction = self.score_model.predict(rgb_array)
        # Remove colors classified as 0 (Other colors)
        prediction = prediction[prediction != 0]
        score = np.average(prediction)
        return score

    def get_image_score_by_year(self, year_range):
        img_helper = imageHelper(self.image_score_filename)
        if img_helper.imageExists():
            image_score_by_year_df = pd.read_csv(self.image_score_filename, index_col=0)
            image_score_by_year = image_score_by_year_df.Score
            return image_score_by_year

        ind = 0
        image_scores = []
        for year in year_range:
            for week in range(1, 52):
                imagePath = self.satellite_dir + 'map_{0}_y{1}_w{2}.png'.format(self.country, year, week)
                img_helper = imageHelper(imagePath)
                if img_helper.imageExists():
                    img_helper.imageSize = self.satellite_image_size
                    score = self.get_image_score(img_helper)
                    image_scores.append([self.country, year, week, score])
                    ind += 1
        imageScoresDf = pd.DataFrame(image_scores)
        imageScoresDf.columns = ["Country", "Year", "Week", "Score"]
        imageScoreByYear = imageScoresDf.groupby("Year")["Score"].mean()
        imageScoreWithoutIndex = imageScoreByYear.reset_index()
        imageScoreWithoutIndex.to_csv(self.image_score_filename, index=False)
        return imageScoreByYear