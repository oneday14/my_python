# 데이터 불러오기
from sklearn.datasets import load_iris
iris_dataset = load_iris() 
X = iris_dataset.data
Y = iris_dataset.target

import pandas as pd
df = pd.DataFrame(X) 

# EDA 자동화 라이브러리
## 1. Sweetviz

import sweetviz as sv

advert_report = sv.analyze([df[0:100], 'train'])  # sv.analyze([dataframe, 이름])
advert_report.show_html('./sweetviz_iris.html')

advert_report2 = sv.compare([df[0:100], 'train'], [df[100:150], 'test'])
advert_report2.show_html('./sweetviz_iris2.html')

## 2. Pandas_profiling

import pandas_profiling

profile = df.profile_report()
profile.to_file(output_file="./profiling_iris.html")
