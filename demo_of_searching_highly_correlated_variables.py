# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd

import variable_selection_based_on_r

threshold_of_r = 0.95  # variable whose absolute correlation coefficnent with other variables is higher than threshold_of_r is searched
threshold_of_rate_of_same_value = 1

# load data set
dataset = pd.read_csv('descriptors_with_logS.csv', encoding='SHIFT-JIS', index_col=0)

dataset = dataset.loc[:, dataset.mean().index]  # 平均を計算できる変数だけ選択
dataset = dataset.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
dataset = dataset.dropna(axis=1)  # nanのある変数を削除

x = dataset.iloc[:, 1:]

# delete variables with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in x.columns:
    num += 1
    #    print('{0} / {1}'.format(num, x.shape[1]))
    same_value_number = x[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where(x.var() == 0)
"""

if len(deleting_variable_numbers[0]) != 0:
    x = x.drop(x.columns[deleting_variable_numbers], axis=1)

print('# of X-variables: {0}'.format(x.shape[1]))

highly_correlated_variable_numbers = variable_selection_based_on_r.search_highly_correlated_variables(x, threshold_of_r)
print('# of highly correlated X-variables: {0}'.format(len(highly_correlated_variable_numbers)))

x_selected = x.drop(x.columns[highly_correlated_variable_numbers], axis=1)
print('# of selected X-variables: {0}'.format(x_selected.shape[1]))
