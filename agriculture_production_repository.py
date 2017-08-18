import pandas as pd


class agriculture_production_repository:
    def __init__(self,file:str):
        self.file = file

    def read_df(self):
        agriculture_production_df = pd.read_csv(self.file)
        return agriculture_production_df