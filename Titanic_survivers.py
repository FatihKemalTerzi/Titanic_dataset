import pandas as pd
import numpy as np
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt



df = seaborn.load_dataset('titanic')

df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    :param dataframe: titanic dataset taken from seaborn library
    :param cat_th: threshold value for categorik variables
    :param car_th: threshold value for cardinal variables
    :return: cat_cols:categorik columns in dataset
             num_cols:nümeric columns in dataset
             cat_but_car: seems like categorik but actually cardinal variables
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =='O']

    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_but_cat = [col for col in num_cols if dataframe[col].nunique()<cat_th]
    cat_cols = num_but_cat +cat_cols

    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols,num_cols,cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

