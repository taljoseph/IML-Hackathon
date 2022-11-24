import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import datetime

GRID_SIZE = 15

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_dict_rev = {'BATTERY' : 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
out_str = {"STREET", "SIDEWALK", "PARKING", "ALLEY", "VEHICLE", "STATION",
            "TRAIN", "BUS", "DRIVEWAY", "PARK", "PLATFORM", "LAND", "LOT", "ATM", "SITE", "TRANSPORTATION",
            "RAILROAD", "HIGHWAY", "WATERFRONT", "TAXICAB", "BRIDGE", "MACHINE", "NEWSSTAND", "FOREST", "CEMETARY", "TRACKS"}
in_public_str = {"STORE", "RESTAURANT", "HOTEL", "HOSPITAL", "BANK", "FACILITY",
                  "BAR", "WORSHIP", "CURRENCY", "AIRPORT", "GOVERNMENT", "SCHOOL",
                  "DEALERSHIP", "LIBRARY", "WASH", "BARBERSHOP", "BOWLING ALLEY", "ABANDONED BUILDING", "SHOP"
                  ,"THEATER", "CREDIT", }
in_private_str = {"APARTMENT", "RESIDENCE", "OFFICE", "HOME", "CLUB", "FACTORY", "WAREHOUSE",
                   "DAY CARE", "FEDERAL", "AIRCRAFT", "UNIVERSITY", "COLLEGE", "JAIL", "ARENA", "POOL", "KENNEL",
                  }

class Bagger:
    """
    The model
    """

    train_data = None

    morning_bagger = None
    noon_bagger = None
    night_bagger = None

    morning_grid = None
    noon_grid = None
    night_grid = None

    morn_x_max = None
    morn_x_min = None
    morn_y_max = None
    morn_y_min = None

    noon_x_max = None
    noon_x_min = None
    noon_y_max = None
    noon_y_min = None

    night_x_max = None
    night_x_min = None
    night_y_max = None
    night_y_min = None

    def __init__(self, train_data):
        """
        Constructor
        :param train_data: The training data
        """

        self.train_data = train_data
        self.train_data.dropna(inplace=True)
        self.preprocess_train_data(train_data)

    def preprocess_train_data(self, train_data):
        """
        Initializes all the attributes of the class, and class the
        preprocessing method on the training data. Also creates our learning
        model
        :param train_data: the train data that was given
        """

        train_data = train_data.dropna()
        df_morning, df_noon, df_night = preprocess(train_data)
        self.morn_x_max = df_morning["X Coordinate"].max()
        self.morn_x_min = df_morning["X Coordinate"].min()
        self.morn_y_max = df_morning["Y Coordinate"].max()
        self.morn_y_min = df_morning["Y Coordinate"].min()

        self.noon_x_max = df_noon["X Coordinate"].max()
        self.noon_x_min = df_noon["X Coordinate"].min()
        self.noon_y_max = df_noon["Y Coordinate"].max()
        self.noon_y_min = df_noon["Y Coordinate"].min()

        self.night_x_max = df_night["X Coordinate"].max()
        self.night_x_min = df_night["X Coordinate"].min()
        self.night_y_max = df_night["Y Coordinate"].max()
        self.night_y_min = df_night["Y Coordinate"].min()

        df_morning, self.morning_grid = self.adding_grid(df_morning, df_morning["X Coordinate"].max(),
                                                    df_morning["X Coordinate"].min(),
                                                    df_morning["Y Coordinate"].max(), df_morning["Y Coordinate"].min())

        df_noon, self.noon_grid = self.adding_grid(df_noon, df_noon["X Coordinate"].max(), df_noon["X Coordinate"].min(),
                                         df_noon["Y Coordinate"].max(), df_noon["Y Coordinate"].min())

        df_night, self.night_grid = self.adding_grid(df_night, df_night["X Coordinate"].max(),
                                                df_night["X Coordinate"].min(), df_night["Y Coordinate"].max(),
                                                df_night["Y Coordinate"].min())

        df_morning = df_morning.drop(["Block"], axis=1)
        df_noon = df_noon.drop(["Block"], axis=1)
        df_night = df_night.drop(["Block"], axis=1)
        morning_df, morning_hat = split(df_morning)
        noon_df, noon_hat = split(df_noon)
        night_df, night_hat = split(df_night)
        max_features = [morning_df.shape[1], noon_df.shape[1], night_df.shape[1]]

        self.train_committee(80, max_features, 15, 13, morning_df, morning_hat, noon_df, noon_hat, night_df, night_hat)

    def train_committee(self, T, max_features, max_depth, min_samples_leaf, morning_df, morning_hat, noon_df,
                        noon_hat, night_df, night_hat):
        """
        Initializes the learning model, and performs fit on the given training
        data that was received
        :param T: amount of learners in committee
        :param max_features: the max features that learner can use
        :param max_depth: the max depth of each tree
        :param min_samples_leaf: the minimum samples leaf
        :param morning_df: the morning dataframe
        :param morning_hat: the morning response dataframe
        :param noon_df: the noon dataframe
        :param noon_hat: the noon response dataframe
        :param night_df: the night dataframe
        :param night_hat: the noon response dataframe
        """
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

        self.morning_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                       min_samples_leaf=min_samples_leaf,
                                                                       class_weight=class_weights,
                                                                       max_features=max_features[0],
                                                                       min_samples_split=50),
                                                T, max_samples=morning_df.shape[0],
                                                max_features=max_features[0],
                                                bootstrap=True).fit(morning_df, morning_hat)

        self.noon_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                    min_samples_leaf=min_samples_leaf,
                                                                    class_weight=class_weights,
                                                                    max_features=max_features[1], min_samples_split=50),
                                             T,  max_samples=noon_df.shape[0], max_features=max_features[1],
                                             bootstrap=True).fit(noon_df, noon_hat)

        self.night_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                     min_samples_leaf=min_samples_leaf,
                                                                     class_weight=class_weights,
                                                                     max_features=max_features[2],
                                                                     min_samples_split=50),
                                              T,  max_samples=night_df.shape[0],
                                              max_features=max_features[2], bootstrap=True).fit(night_df, night_hat)


    def grid_for_test(self, df, x_max, x_min, y_max, y_min, dicten):
        """
        Updates dataframe's x, y to categorical x and y features by grid.
        :param df: Test dataframe
        :param x_max: maximum x
        :param x_min: minimum x
        :param y_max: maximum y
        :param y_min: minimum y
        :param dicten: dictionary of categorical variables names.
        :return: updated dataframe
        """
        keys = list(dicten.keys())

        df.reset_index(drop=True, inplace=True)

        for key in keys:
            zeros = np.zeros((df.shape[0], 1))
            str_key = str(key)
            zeros = pd.DataFrame(zeros.astype(int), columns=[str_key], index=None)
            df = pd.concat([df, zeros], axis=1)

        X_cor = np.array(((df["X Coordinate"] - x_min) / (x_max - x_min)) * (GRID_SIZE - 1)).astype(int)
        Y_cor = np.array(((df["Y Coordinate"] - y_min) / (y_max - y_min)) * (GRID_SIZE - 1)).astype(int)
        del df["X Coordinate"]
        del df["Y Coordinate"]
        cor = list(zip(X_cor, Y_cor))
        for index in range(df.shape[0]):
            if str(cor[index]) in keys:
                df.loc[index, dicten[str(cor[index])]] = 1
        return df

    def adding_grid(self, df, x_max, x_min, y_max, y_min):
        """
        Creates a grid according to training dataframe, x, y values and GRID_SIZE
        to change x,y values to regional area representing categorical features.
        Only areas with a certain number of samples are valid.
        :param df: Training dataframe.
        :param x_max: maximum value of x
        :param x_min: minimum value of x
        :param y_max: maximum value of y
        :param y_min: minimum value of y
        :return: returns the updated dataframe and some dictionary
        """
        X_cor = np.array(((df["X Coordinate"] - x_min) / (x_max - x_min)) * (GRID_SIZE - 1)).astype(int)
        Y_cor = np.array(((df["Y Coordinate"] - y_min) / (y_max - y_min)) * (GRID_SIZE - 1)).astype(int)
        cor = set(zip(X_cor, Y_cor))
        dic = {}
        for cord in cor:
            dic[cord] = []
        for index in range(df.shape[0] - 1):
            key = (X_cor[index], Y_cor[index])
            if key in dic.keys():
                dic[key].append(index)
        dicten = dic
        keys = list(dic.keys())
        del df["X Coordinate"]
        del df["Y Coordinate"]
        df.reset_index(drop=True, inplace=True)
        for key in keys:
            zeros = np.zeros((df.shape[0], 1))
            zeros[dic[key]] = 1
            str_key = str(key)
            zeros = pd.DataFrame(zeros.astype(int), columns=[str_key], index=None)
            df = pd.concat([df, zeros], axis=1)
        return df, dicten


def preprocess(df):
    """
    Pre-processes the given data, getting rid of unnecessary columns,
    as well as adjusts certain features and splits the data into 3
    based on the time of the day.
    :param df: The dataframe
    :return: 3 dataframes, each of samples corresponding to different
    times of the day (morning, noon and night).
    """
    cols = df.columns
    if "Unnamed: 0" in cols:
        del df["Unnamed: 0"]
    if "Unnamed: 0.1" in cols:
        del df["Unnamed: 0.1"]
    del df["ID"]
    del df["Case Number"]
    del df["Year"]
    del df["Updated On"]
    del df["IUCR"]
    del df["FBI Code"]
    del df["Description"]
    del df["Latitude"]
    del df["Longitude"]
    del df["Location"]

    if "Primary Type" in cols:
        df.replace({"Primary Type": crimes_dict_rev}, inplace=True)

    outside = df["Location Description"].to_frame()
    in_private = df["Location Description"].to_frame()
    in_public = df["Location Description"].to_frame()
    outside.rename(columns={"Location Description": "out"}, inplace=True)
    in_private.rename(columns={"Location Description": "in_private"}, inplace=True)
    in_public.rename(columns={"Location Description": "in_public"}, inplace=True)
    outside = outside.out.str.contains("|".join(out_str)).to_frame()
    in_private = in_private.in_private.str.contains("|".join(in_private_str)).to_frame()
    in_public = in_public.in_public.str.contains("|".join(in_public_str)).to_frame()
    df = pd.concat([df, outside, in_private, in_public], axis=1)
    date_copy = df["Date"].apply(lambda x: x[:10])
    phi_two = (2 * np.pi) / 7
    date_copy_two = date_copy.apply(lambda x: datetime.date(int(x[7:]), int(x[0:2]), int(x[3:5])).weekday())
    df["d1"] = date_copy_two.apply(lambda x: np.cos(phi_two * x))
    df["d2"] = date_copy_two.apply(lambda x: np.sin(phi_two * x))
    time_2 = df["Date"].apply(lambda x: int(x[11:13]) + (int(x[14:16]) / 60) + (int(x[17:19]) / 3600))
    phi = (2 * np.pi) / 24
    df["X Time"] = time_2.apply(lambda x: x * np.cos(phi))
    df["Y Time"] = time_2.apply(lambda x: x * np.sin(phi))
    del df["Location Description"]
    time = df["Date"].apply(lambda x: int(x[11:13]) if x[20:] == "AM" else int(x[11:13]) + 12)
    del df["Date"]
    df = df.join(time)
    df.rename(columns={"Date": "Time"}, inplace=True)
    morning = df[(df['Time'] >= 6) & (df['Time'] < 14)]
    noon = df[(df['Time'] >= 14) & (df['Time'] < 22)]
    night = df[((df['Time'] >= 22) & (df['Time'] <= 24) |
                (df['Time'] >= 0) & (df['Time'] < 6))]
    del morning["Time"]
    del noon["Time"]
    del night["Time"]
    return morning, noon, night

def split(df: pd.DataFrame):
    """
    Splits the dataframe into the features and the response vector
    :param df: The dataframe
    :return: Matrix X of features and vector y of the corresponding responses
    """
    y = df["Primary Type"]
    X = df.drop(["Primary Type"], axis=1)
    return X.to_numpy(), y.to_numpy()
