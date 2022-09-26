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


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

#cat
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "survived")


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

##################################