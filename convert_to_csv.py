# -*- coding: utf-8 -*-

from google.colab import files

file = files.upload()

print(len(file.keys()))

print(file)

import pandas

import io 

dataframe = pandas.read_csv(io.StringIO(file["author_and_quote.csv"].decode("utf-8")), 
                            sep=",")

print(dataframe.head().to_string())

print(dataframe.head())

print(dataframe.describe())

quotes_list = dataframe["Quote"].tolist()

print(quotes_list)
