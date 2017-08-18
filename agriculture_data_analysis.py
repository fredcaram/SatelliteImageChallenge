import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from agriculture_production_repository import agriculture_production_repository
from image_score_repository import image_score_repository


class AgricultureDataAnalysis:
    def __init__(self, pixels_to_remove, score_model, satellite_image_size: list):
        self.pixels_to_remove = pixels_to_remove
        self.score_model = score_model
        self.satellite_image_size = satellite_image_size
        self.start_year = 1997

        agriculture_data_file = "FAOSTAT\\FAOSTAT_data_8-3-2017.csv"
        agriculture_repo = agriculture_production_repository(agriculture_data_file)
        self.agriculture_df = agriculture_repo.read_df()

        year_range = range(self.start_year, 2013 + 1)
        img_repo = image_score_repository(self.score_model,
                                          self.pixels_to_remove, self.satellite_image_size)
        self.image_score_by_year = img_repo.get_image_score_by_year(year_range)

    def general_average_analysis(self):
        # Got 3 year before to use for the rolling mean
        # Just Net Production
        element_code = 154

        agriculture_production_value = self.__get_normalized_value_by_year__(element_code)

        # The rolling mean was used to represent the growing trend
        agricultureProductionRollingMean = agriculture_production_value.rolling(window=2, center=False).mean()

        # Plot productionXscore plot
        agricultureProductionValueFrom2000 = agriculture_production_value[agriculture_production_value.index >= 2000]
        agricultureProductionRollingMeanFrom2000 = agricultureProductionRollingMean[
            agricultureProductionRollingMean.index >= 2000]
        agricultureProductionNormalizedFrom2000 = agricultureProductionValueFrom2000 - agricultureProductionRollingMeanFrom2000
        prodAndImageCov = np.corrcoef(
            agricultureProductionNormalizedFrom2000[agricultureProductionNormalizedFrom2000.index <= 2013].values,
            self.image_score_by_year[self.image_score_by_year.index <= 2013].values)

        fig, axes = plt.subplots(3, figsize=(8, 6), sharex=True)
        # When comparing the production - rolling mean with the image greenish, a negative correlation was found (~0.44, ~0.42)
        #  when using window = 3 and = 2
        #print(prodAndImageCov)
        #plt.text(0, 0, 'Correlation = {0}'.format(prodAndImageCov))
        agricultureProductionValueFrom2000.plot(ax=axes[0], color='blue')
        agricultureProductionRollingMeanFrom2000.plot(ax=axes[0], color='red', title="Normalized value by year with trend")
        agricultureProductionNormalizedFrom2000.plot(ax=axes[1], title="Normalized value by year without trend")
        self.image_score_by_year.plot(ax=axes[2], title="Image greeness by year")
        plt.show()

    def __get_normalized_value_by_year__(self, element_code):
        agriculture_df = self.agriculture_df.loc[self.agriculture_df["Year"] >= self.start_year,]
        agriculture_df = agriculture_df.loc[agriculture_df["Element Code"] == element_code]
        agriculture_df = agriculture_df.sort_values("Year")
        agriculture_df_mean_by_item = agriculture_df.groupby("Item")["Value"].agg([np.mean, np.std])
        agriculture_df_mean_by_item = agriculture_df_mean_by_item.reset_index()
        agriculture_df = agriculture_df.merge(agriculture_df_mean_by_item, left_on="Item", right_on="Item")
        agriculture_df["NormValue"] = (agriculture_df["Value"] - agriculture_df["mean"]) / agriculture_df["std"]
        agriculture_production_value = agriculture_df.groupby("Year")["NormValue"].mean()
        return agriculture_production_value

    def analyze_performance_per_item(self):
        agricultureByYearAndItem2000 = self.__get_agriculture_by_year_and_item__(2000)
        aggriculturePerItemCorrelation = self.__get_correlation_per_item__(agricultureByYearAndItem2000)

        meltedAgricultureDf = pd.melt(agricultureByYearAndItem2000, id_vars=['Item', 'Year'],
                                      value_vars=['Score', 'NormalizedValue'])

        print("Top 10 negative correlations")
        self.__display_top_n_correlations__(aggriculturePerItemCorrelation, meltedAgricultureDf, 10, True)

        print("Top 10 positive correlations")
        self.__display_top_n_correlations__(aggriculturePerItemCorrelation, meltedAgricultureDf, 10, False)

    def __get_correlation_per_item__(self, agricultureByYearAndItem2000):
        itemCorrelations = []
        items = np.unique(agricultureByYearAndItem2000["Item"].values)
        for item in items:
            agriculturePerItem = agricultureByYearAndItem2000[agricultureByYearAndItem2000["Item"] == item]
            corr = np.corrcoef(
                agriculturePerItem["NormalizedValue"].values,
                agriculturePerItem["Score"].values)
            itemCorrelations.append(corr[0, 1])
        aggriculturePerItemCorrelation = pd.DataFrame({"Item": items, "Correlation": itemCorrelations})
        return aggriculturePerItemCorrelation

    def __get_agriculture_by_year_and_item__(self, start_date: int):
        agricultureByYearAndItem = self.agriculture_df.groupby(["Year", "Item"])["Value"].mean()

        agricultureByYearAndItemRollingMean = agricultureByYearAndItem.rolling(window=3, center=False).mean()
        agricultureByYearAndItemRollingMean.name = "RollingMean"
        agricultureByYearAndItemRollingMean = agricultureByYearAndItemRollingMean.reset_index()
        agricultureByYearAndItemRollingMean2000 = agricultureByYearAndItemRollingMean[
            agricultureByYearAndItemRollingMean["Year"] >= start_date]

        agricultureByYearAndItem = agricultureByYearAndItem.reset_index()
        agricultureByYearAndItem2000 = agricultureByYearAndItem[agricultureByYearAndItem["Year"] >= start_date]
        agricultureByYearAndItem2000 = agricultureByYearAndItem2000.merge(agricultureByYearAndItemRollingMean2000,
                                                                          left_on=["Year", "Item"],
                                                                          right_on=["Year", "Item"])
        agricultureByYearAndItem2000["NormalizedValue"] = agricultureByYearAndItem2000["Value"] - \
                                                          agricultureByYearAndItem2000["RollingMean"]
        image_score_by_year = self.image_score_by_year.reset_index()
        agricultureByYearAndItem2000 = agricultureByYearAndItem2000.merge(image_score_by_year, left_on="Year",
                                                                          right_on="Year")
        return agricultureByYearAndItem2000

    def __display_top_n_correlations__(self, aggriculturePerItemCorrelation,
                                       meltedAgricultureDf, top_n=5, ascending=True):
        topcorr = aggriculturePerItemCorrelation.sort_values("Correlation", ascending=ascending).head(top_n)
        topCorrAgricultureDf = meltedAgricultureDf.merge(topcorr, how="inner", left_on="Item",
                                                         right_on="Item")
        topCorrAgricultureDf = topCorrAgricultureDf.sort_values("Correlation")
        print(topcorr)
        #g1 = sns.FacetGrid(topCorrAgricultureDf, col="variable", row='Item', sharex=True, sharey=False,
        #                   size=3, aspect=2)
        #g1 = g1.map(sns.pointplot, "Year", "value")
        #plt.show()
        return topCorrAgricultureDf
