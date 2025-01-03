import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

class ElectricityDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        self.df_test = pd.read_csv(self.file_path)
        self.df_test["2016/9/18"] = [np.nan] * len(self.df_test)
        self.df_date = self.df_test.iloc[:, 1:]
        self.df_other = self.df_test.iloc[:, :1]

    def preprocess_dates(self):
        self.df_date.set_index(self.df_date.columns[0], inplace=True)
        self.df_date.columns = pd.to_datetime(self.df_date.columns)
        self.df_date = self.df_date.T
        self.df_date = self.df_date.sort_index()
        self.df_date = self.df_date.T.reset_index()

        self.df_date2 = self.df_date.T
        self.df_date2.index = pd.to_datetime(self.df_date2.index)
        self.df_date2['weekday'] = self.df_date2.index.weekday

        weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        self.df_date2['weekday'] = self.df_date2['weekday'].map(weekday_map)

        new_row = pd.DataFrame(columns=self.df_date2.columns, index=[self.df_date2.index.min() - timedelta(days=1)])
        new_row['weekday'] = 'Tuesday'

        self.df_date2 = pd.concat([new_row, self.df_date2])
        self.df_date2.index = pd.to_datetime(self.df_date2.index)

    def split_by_year(self):
        self.df_date2_2014 = self.df_date2.loc['2014']
        self.df_date2_2015 = self.df_date2.loc['2015']
        self.df_date2_2016 = self.df_date2.loc['2016']

    def recover_missing_data(self, df):
        months = df.index.to_period("M").unique()
        month_data_recov = {}

        for month in months:
            df_month = df.loc[str(month)]
            df_recov = df_month.copy()

            for column in df_month.columns[:-1]:
                for day in df_month['weekday'].unique():
                    day_data = df_month.loc[df_month['weekday'] == day, column]

                    if day_data.isna().all():
                        df_recov.loc[df_month['weekday'] == day, column] = -1
                    elif day_data.isna().any():
                        mean_val = round(day_data.mean(), 2)
                        df_recov.loc[df_month['weekday'] == day, column] = day_data.fillna(mean_val)

            if not df_recov.empty:
                month_data_recov[str(month)] = df_recov

        if month_data_recov:
            df_recov_all = pd.concat(month_data_recov.values(), axis=0)
            df_recov_all.reset_index(drop=False, inplace=True)
        else:
            df_recov_all = pd.DataFrame()

        return df_recov_all

    def recover_all_years(self):
        self.df_date2_2014_recov = self.recover_missing_data(self.df_date2_2014)
        self.df_date2_2015_recov = self.recover_missing_data(self.df_date2_2015)
        self.df_date2_2016_recov = self.recover_missing_data(self.df_date2_2016)

        self.df_date2_all_recov_week = pd.concat([self.df_date2_2014_recov, self.df_date2_2015_recov, self.df_date2_2016_recov], axis=0, ignore_index=False)
        if 'weekday' in self.df_date2_all_recov_week.columns:
            self.df_date2_all_recov = self.df_date2_all_recov_week.drop('weekday', axis=1)

        self.df_date2_all_recov.set_index('index', inplace=True)
        self.df_date2_all_recov.index.name = None
        self.df_date_all_recov = self.df_date2_all_recov.T
        self.df_test_all_recov = pd.concat([self.df_other, self.df_date_all_recov], axis=1)

    def missing_value(self):
        self.load_data()
        self.preprocess_dates()
        self.split_by_year()
        self.recover_all_years()
        return self.df_test_all_recov

    def preprocess_data(self):
        self.df_test_all_recov = self.new_preprocess_data(self.df_test_all_recov)
        return self.df_test_all_recov

    @staticmethod
    def new_preprocess_data(df):

        def winsorize_dataframe(df, lower_percentile=0.05, upper_percentile=0.95):
            return df.apply(lambda x: np.clip(x, x.quantile(lower_percentile), x.quantile(upper_percentile)), axis=1)

        df.iloc[:, 1:] = winsorize_dataframe(df.iloc[:, 1:])

        def min_max_scale_dataframe(df):
            min_val = df.min().min()
            max_val = df.max().max()
            df_scaled = (df - min_val) / (max_val - min_val)
            return df_scaled

        df.iloc[:, 1:] = min_max_scale_dataframe(df.iloc[:, 1:])
        return df

    def detect_anomalies(self, model_path):
        self.df_new = self.preprocess_data()
        X_new = self.df_new.drop(columns=['CONS_NO'])
        clof = joblib.load(model_path)
        y_new_scores = clof.decision_function(X_new)
        self.df_new['anomaly_score'] = y_new_scores

        new_anomalies_top_50 = self.df_new.sort_values(by='anomaly_score', ascending=False).head(50)
        for cons_no, anomaly_score in zip(new_anomalies_top_50['CONS_NO'], new_anomalies_top_50['anomaly_score']):
            print(f'CONS_NO: {cons_no}, CLOF Score: {anomaly_score}')

    def detect_anomalies(self, model_path):
            X_new = self.df_test_all_recov.drop(columns=['CONS_NO'])
            clof = joblib.load(model_path)
            y_new_scores = clof.decision_function(X_new)
            self.df_test_all_recov['anomaly_score'] = y_new_scores

            new_anomalies_top_50 = self.df_test_all_recov.sort_values(by='anomaly_score', ascending=False).head(50)

            return new_anomalies_top_50[['CONS_NO', 'anomaly_score']]

if __name__ == '__main__':
    # 定義數據文件路徑和模型文件路徑
    test_file_path = '/Users/ranli/Documents/python_ve/MS_BDA/data/electricitytheft_test_2.csv'
    model_file_path = '/Users/ranli/Documents/python_ve/MS_BDA/weekly_nan_test/cblof_model_2.joblib'
    save_path ='/Users/ranli/Documents/python_ve/MS_BDA/weekly_nan_test'

    # missing value
    processor = ElectricityDataProcessor(test_file_path)
    recover_df = processor.missing_value()  
    recover_df.to_csv(f'{save_path}/recovered_data.csv', index=False)

    # detect anomalies (clof)
    detect_df = processor.preprocess_data()
    anomalies =processor.detect_anomalies(model_file_path)
    print(anomalies)


# 如果要用在其他 py （ipynb）檔
# from cblof_preprocess import ElectricityDataProcessor

