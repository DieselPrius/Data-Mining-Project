from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz
import pickle
from sklearn.neighbors import KNeighborsClassifier
import random
import math


#train_2016_v2_df = pd.read_csv("train_2016_v2.csv")
#properties_2016_df = pd.read_csv("properties_2016.csv")
train_2017_df = pd.read_csv("train_2017.csv")
properties_2017_df = pd.read_csv("properties_2017.csv")
#zillow_data_dictionary = pd.read_excel("zillow_data_dictionary.xlsx","Data Dictionary", index_col=None)

#unpartitioned error stats
print(train_2017_df)
print("mean = ",end='')
print(train_2017_df.loc[:,"logerror"].mean())
print("median = ",end='')
print(train_2017_df["logerror"].median())
print("mode = ")
print(train_2017_df["logerror"].mode())
print("sd = ",end='')
print(train_2017_df["logerror"].std())
print("max = ",end='')
print(train_2017_df["logerror"].max())
print("min = ",end='')
print(train_2017_df["logerror"].min())
print(train_2017_df["logerror"].quantile([.25,.5,.75]))



train_2017_df["logerror"].hist(bins=700)


#analyze features without partitioning:
#print(train_2017_df.columns)
merged_2017 = pd.merge(train_2017_df, properties_2017_df, on="parcelid")
print(merged_2017.head)


#create the perc missing data frame:
colName = []
missingPerc = []
totalNumOfRecords = len(merged_2017.index)
for column in merged_2017:
    colName.append(column)
    percMissing = (merged_2017[column].isnull().sum()/totalNumOfRecords) * 100
    missingPerc.append(percMissing)
d = {'Feature': colName, 'perc_missing': missingPerc}
perc_missing_df = pd.DataFrame(data=d)
print(perc_missing_df)
#perc_missing_df = pd.merge(perc_missing_df, zillow_data_dictionary, on="Feature", how="left")
print(perc_missing_df)
perc_missing_df = perc_missing_df.sort_values(by="perc_missing",ascending=False)
writer = pd.ExcelWriter('missing_vals.xlsx')
perc_missing_df.to_excel(writer,'Sheet1')
writer.save()


#analysis needed for feature selection
merged_2017.plot.scatter(y="storytypeid",x="logerror")
merged_2017.plot.scatter(y="yardbuildingsqft26",x="logerror")
merged_2017["fireplaceflag"] = pd.to_numeric(merged_2017["fireplaceflag"])
merged_2017.plot.scatter(y="fireplaceflag",x="logerror")
merged_2017.plot.scatter(y="architecturalstyletypeid",x="logerror")

merged_2017.plot.scatter(y="fips",x="logerror")
#merged_2017["propertycountylandusecode"] = pd.to_numeric(merged_2017["propertycountylandusecode"])
#merged_2017.plot.scatter(y="propertycountylandusecode",x="logerror")
merged_2017.plot.scatter(y="propertylandusetypeid",x="logerror")
merged_2017.plot.scatter(y="rawcensustractandblock",x="logerror")
merged_2017.plot.scatter(y="roomcnt",x="logerror")

##merged_2017.fillna(-1, inplace=True)

#distribution of assessment year:
#merged_2017["assessmentyear"].hist() #looks the same for all rows, recommend removing
#merged_2017["bathroomcnt"].hist(bins=20)
#merged_2017["fips"].hist()
#merged_2017.plot.scatter(y="calculatedfinishedsquarefeet",x="logerror")
#merged_2017.plot.scatter(x="latitude",y="longitude")
#merged_2017.plot.scatter(x="logerror",y="poolcnt")


#partition the data
print(merged_2017[(merged_2017["logerror"] > -0.001) & (merged_2017["logerror"] < 0.001)]["logerror"])


#merged_2017["bedroomcnt"].plot.bar()    

pylab.show()