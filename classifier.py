import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
import math
import random
import pickle
from bagger import Bagger


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
NIGHT_MIN_X_CORD = 1092706.0
NIGHT_MAX_X_CORD = 1205112.0
NIGHT_MIN_Y_CORD = 1813910.0
NIGHT_MAX_Y_CORD = 1951493.0

# NIGHT_MIN_X_CORD = 1092706.0
# NIGHT_MAX_X_CORD = 1205112.0
# NIGHT_MIN_Y_CORD = 1813910.0
# NIGHT_MAX_Y_CORD = 1951493.0



def send_police_cars(X):
    """
    Upon given a date, the function learns from the given date and previous
    data and predicts where should we send 30 cars throughout the day in order
    to prevents as much crime as possible!
    :param X: Date in correct form.
    :return: List of 30 tuples of (x, y, time) spot and time to place a police
    car for 30 minutes.
    """
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)

    lst = []
    for item in X:
        df = model.train_data
        original_date = item[:11]
        month, day, year = int(item[:2]), int(item[3:5]), int(item[6:10])
        df = df.loc[:, ["X Coordinate", "Y Coordinate", "Date"]]
        df.dropna(inplace=True)

        date_df = df["Date"].apply(lambda x: x[:10])
        date_df = date_df.apply(lambda x: datetime.date(int(x[6:10]), int(x[0:2]), int(x[3:5])).weekday())
        time = df["Date"].apply(lambda x: int(x[11:13]) if x[20:] == "AM" else int(x[11:13]) + 12)
        del df["Date"]
        df = df.join(time)
        df.rename(columns={"Date": "Time"}, inplace=True)
        df = df.join(date_df)
        df.rename({"Date": "Weekday"}, axis=1, inplace=True)
        weekday = datetime.date(year, month, day).weekday()
        df = df[df["Weekday"] == weekday]
        df.drop("Weekday", inplace=True, axis=1)
        k_means = KMeans(30).fit(df.to_numpy())

        def fun(x):
            """
            fun function - changes a numeric value to time with date prefix.
            """
            x = str(x)
            i = x.find('.')
            if i == -1:
                x2 = "00"
            else:
                x2 = math.floor(float(x[2:]) * 60)
                if x2 >= 10:
                    x2 -= 10
                    x2 = str(x2)
                    if len(x2) == 1:
                        x2 = "0" + x2
                else:
                    x2 = "00"
                x = x[:2]
            x1 = x
            x3 = "00"
            return original_date + x1 + ":" + x2 + ":" + x3

        time = np.array([fun(x) for x in k_means.cluster_centers_[:, -1]])
        xyt = k_means.cluster_centers_[:, : -1]
        time = time.reshape((30, 1))
        xyt = np.hstack((xyt, time))
        lst.append(list(map(tuple, xyt)))
    return lst


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


def save_pickle():
    """
    Creates a pickle file containing the class data fitted over a training
    data
    """
    df = pd.read_csv("Dataset_crimes.csv")
    df_extra = pd.read_csv("crimes_dataset_part2.csv")
    df_extra.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_extra], axis=0)
    model = Bagger(df)
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def split(df: pd.DataFrame):
    """
    Splits the dataframe into the features and the response vector
    :param df: The dataframe
    :return: Matrix X of features and vector y of the corresponding responses
    """
    y = df["Primary Type"]
    X = df.drop(["Primary Type"], axis=1)
    return X.to_numpy(), y.to_numpy()


def coordinate_update(df, df_nan, index, row, coordinate):
    """
    Updates coordinates given the axis aren't present in the sample data
    :param df: The dataframe of the trained model with no NaN values
    :param df_nan: The dataframe of the test data
    :param index: The index of the sample in the df_nan
    :param row: The row itself of the index in df_nan
    :param coordinate: On which coordinate to operate
    :return: None
    """
    if not pd.isnull(row["Block"]):
        frame = df[df["Block"] == row["Block"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, coordinate] = frame.at[random.randint(0, frame.shape[0] - 1), coordinate]
            return
    if not pd.isnull(row["Beat"]):
        frame = df[df["Beat"] == row["Beat"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, coordinate] = frame.at[random.randint(0, frame.shape[0] - 1), coordinate]
            return
    if not pd.isnull(row["District"]):
        frame = df[df["District"] == row["District"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, coordinate] = frame.at[random.randint(0, frame.shape[0] - 1), coordinate]
            return
    if not pd.isnull(row["Ward"]):
        frame = df[df["Ward"] == row["Ward"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, coordinate] = frame.at[random.randint(0, frame.shape[0] - 1), coordinate]
            return
    if not pd.isnull(row["Community Area"]):
        frame = df[df["Community Area"] == row["Community Area"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, coordinate] = frame.at[random.randint(0, frame.shape[0] - 1), coordinate]
            return
    df_nan.at[index, coordinate] = random.randint(df[coordinate].min, df[coordinate].max)


def beat_update(df, df_nan, index, row):
    """
    Re-creates the Beat parameter in the case it's not given in the data
    :param df: The dataframe of the trained model with no NaN values
    :param df_nan: The dataframe of the test data
    :param index: The index of the sample in the df_nan
    :param row: The row itself of the index of df_nan
    :return: None
    """
    if not pd.isnull(row["District"]):
        frame = df[df["Block"] == row["Block"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, "Beat"] = frame.at[random.randint(0, frame.shape[0] - 1), "Beat"]
            return
    if not pd.isnull(row["Ward"]):
        frame = df[df["Ward"] == row["Ward"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, "Beat"] = frame.at[random.randint(0, frame.shape[0] -1), "Beat"]
            return
    if not pd.isnull(row["Community Area"]):
        frame = df[df["Community Area"] == row["Community Area"]].reset_index()
        if frame.shape[0] != 0:
            df_nan.at[index, "Beat"] = frame.at[random.randint(0, frame.shape[0] - 1), "Beat"]
            return
    df_c = df.copy()
    f1 = df_c["X Coordinate"].apply(lambda x: str(x)[:3]).reset_index()
    f2 = df_c["Y Coordinate"].apply(lambda y: str(y)[:3]).reset_index()
    df_c = df_c.reset_index()
    frame = df_c[(f1["X Coordinate"] == str(df_nan.at[index, "X Coordinate"])[:3]) &
               (f2["Y Coordinate"] == str(df_nan.at[index, "Y Coordinate"])[:3])].reset_index()
    if frame.shape[0] != 0:
        df_nan.at[index, "Beat"] = frame.at[random.randint(0, frame.shape[0] - 1), "Beat"]
    else:
        df_nan.at[index, "Beat"] = df.at[random.randint(0, df.shape[0] - 1), "Beat"]


def predict(X):
    """
    Predicts the type of felony:
    0: Battery, 1: Theft, 2: Criminal Damage, 3: Deceptive Practice, 4 :Assault
    :param X: The features matrix
    :return: The predicted response vector, based on the given matrix X
    """
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)
    df = model.train_data  # the original data
    df_X = pd.read_csv(X)
    # code to extract a random time:
    date = df["Date"].apply(lambda x: int(x[11:13]) if x[20:] == "AM" else int(x[11:13]) + 12)
    hours = np.array(date.value_counts().index.tolist())
    hours[hours == 24] = 0
    weights = date.value_counts().to_numpy()
    weights = weights / weights.sum()
    arrest = df["Arrest"].value_counts().to_numpy()
    arrest = arrest / arrest.sum()
    dom = df["Domestic"].value_counts().to_numpy()
    dom = dom / dom.sum()

    # code to extract all rows with NaN
    df_nan = df_X[df_X.isnull().any(axis=1)]
    for index, row in df_nan.iterrows():
        if pd.isnull(row["Date"]):
            r = random.choices(hours, weights, k=1)
            df_nan.at[index, "Date"] = "1/1/2021 {}:00".format(r[0])

        if pd.isnull(row["Location Description"]):
            df_nan.at[index, "Location Description"] = "OTHER"

        if pd.isnull(row["Arrest"]):
            r = random.choices([False, True], arrest)
            df_nan.at[index, "Arrest"] = r[0]

        if pd.isnull(row["Domestic"]):
            r = random.choices([False, True], dom)
            df_nan.at[index, "Domestic"] = r[0]

        if np.isnan(row["X Coordinate"]):
            coordinate_update(df, df_nan, index, row, "X Coordinate")

        if np.isnan(row["Y Coordinate"]):
            coordinate_update(df, df_nan, index, row, "Y Coordinate")

        if pd.isnull(row["Beat"]):
            beat_update(df, df_nan, index, row)

        if pd.isnull(row["District"]):
            size = str(df_nan.at[index, "Beat"])
            df_nan.at[index, "District"] = int(size[:1]) if len(size) == 3 else int(size[:2])

        if pd.isnull(row["Ward"]):
            frame = df[df["District"] == df_nan.at[index, "District"]].reset_index()
            if frame.shape[0] != 0:
                df_nan.at[index, "Ward"] = frame.at[random.randint(0, frame.shape[0] - 1), "Ward"]
            else:
                df_nan.at[index, "Ward"] = df.at[random.randint(0, df.shape[0] - 1), "Ward"]
        if pd.isnull(row["Community Area"]):
            frame = df[df["Ward"] == df_nan.at[index, "Ward"]].reset_index()
            if frame.shape[0] != 0:
                df_nan.at[index, "Community Area"] = \
                    frame.at[random.randint(0, frame.shape[0] - 1), "Community Area"]
            else:
                df_nan.at[index, "Community Area"] = \
                    df.at[random.randint(0, df.shape[0] - 1), "Community Area"]

    df_X.loc[df_nan.index] = df_nan
    cols = df_X.columns
    if "Primary Type" in cols:
        del df_X["Primary Type"]
    del df_X["Block"]
    size = df_X.shape[0]
    df_X.insert(0, "i", np.arange(size))
    morning, noon, night = preprocess(df_X)
    morn_arr = morning["i"].to_numpy()
    noon_arr = noon["i"].to_numpy()
    night_arr = night["i"].to_numpy()
    del morning["i"]
    del noon["i"]
    del night["i"]
    morning = model.grid_for_test(morning, model.morn_x_max, model.morn_x_min, model.morn_y_max,
                        model.morn_y_min, model.morning_grid)
    noon = model.grid_for_test(noon, model.noon_x_max, model.noon_x_min, model.noon_y_max,
                        model.noon_y_min, model.noon_grid)
    night = model.grid_for_test(night, model.night_x_max, model.night_x_min, model.night_y_max,
                        model.night_y_min, model.night_grid)
    final = np.zeros(size)
    if morning.shape[0] != 0:
        morn_final = model.morning_bagger.predict(morning)
        final[morn_arr] = morn_final
    if noon.shape[0] != 0:
        noon_final = model.noon_bagger.predict(noon)
        final[noon_arr] = noon_final
    if night.shape[0] != 0:
        night_final = model.night_bagger.predict(night)
        final[night_arr] = night_final
    return final


def create_files(path):
    """
    Divides the data into 3 sets: Train, Validation and Test
    :param path: The path to the given file (in csv format)
    """
    df = pd.read_csv(path)
    train = df.sample(frac=0.70)
    df = df.drop(train.index)
    test = df.sample(frac=0.10)
    valid = df.drop(test.index)
    test.to_csv("test.csv")
    train.to_csv("train.csv")
    valid.to_csv("validation.csv")
