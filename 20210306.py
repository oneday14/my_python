import numpy as np
import pandas as pd

# 벡터 구조
x1 = np.array([1, 2, 3])
x1

x1.shape

# 매트릭스 구조
x2 = np.array([[1], [2], [3]])
x2

x2.shape

A = np.array([[1, 2, 3],
              [4, 5, 6]])
A

A.reshape(-1, 1).shape      # -1은 알아서 정해짐

B = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

sample_arr = np.reshape(B, (4, 2))
sample_arr

np.concatenate([sample_arr, sample_arr, sample_arr], axis=0).shape  # 합치기
 
DF = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['Var1', 'Var2', 'Var3'])
DF

DF.head(1)
DF.info()
# =============================================================================
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Var1    3 non-null      int64
#  1   Var2    3 non-null      int64
#  2   Var3    3 non-null      int64
# =============================================================================

DF.describe()
# =============================================================================
#        Var1  Var2  Var3
# count   3.0   3.0   3.0
# mean    4.0   5.0   6.0
# std     3.0   3.0   3.0
# min     1.0   2.0   3.0
# 25%     2.5   3.5   4.5
# 50%     4.0   5.0   6.0
# 75%     5.5   6.5   7.5
# max     7.0   8.0   9.0
# =============================================================================

DF.values
 
pd.concat([DF, DF], axis = 0).shape

import sklearn
import matplotlib.pyplot as plt

import os
os.getcwd()

data = pd.read_csv('train_hospital.csv')
label = data['OC']
del data['OC']

data.head()
data.describe()
data.info()

from sklearn.preprocessing import MinMaxScaler
mMscaler = MinMaxScaler()

cat_columns = ['sido', 'instkind', 'ownerChange']
num_columns = [c for c in data.columns if c not in cat_columns]

numeric_data = data[num_columns].values
mMscaler.fit(numeric_data)
mMscalered_data = mMscaler.transform(numeric_data)
mMscalered_data = pd.DataFrame(mMscalered_data, columns = num_columns)

# 결측치 시각화
# !pip install missingno
import missingno as msno
msno.matrix(data)

# 결측치 개수 파악
pd.isna(data).sum()
pd.isna(data).sum().sum()

# 결측치 채우는 방법
# 1. 평균
mean_df = data.copy()

for c in num_columns:
    mean_df.loc[pd.isna(data[c]), c] = data[c].mean()

pd.isna(mean_df[num_columns]).sum().sum()

# 2. 중앙값
median_df = data.copy()

for c in num_columns:
    median_df.loc[pd.isna(data[c]), c] = data[c].median()

pd.isna(median_df[num_columns]).sum().sum()

# 3. Iterative Impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

impute_df = data.copy()

imp_mean = IterativeImputer(random_state=0)
impute_df[num_columns] = imp_mean.fit_transform(impute_df[num_columns])

pd.isna(impute_df[num_columns]).sum().sum()

# 4. 최빈값
mode_df = data.copy()

for c in cat_columns:
    mode_df.loc[pd.isna(data[c]), c] = data[c].mode()[0]

pd.isna(mode_df[cat_columns]).sum().sum()


