# [ 연습 문제 : test score가 가장 높은 매개변수 찾기 ]
run profile1
from sklearn.datasets import load_iris
df_iris = load_iris()

from sklearn.model_selection import cross_val_score

score_te = []
for i in range(2,11) : 
    m_dt = dt(min_samples_split = i, random_state=0) 
    v_score = cross_val_score(m_dt, df_iris.data, df_iris.target, cv=5)
    score_te.append(v_score.mean())
 
# 6. 특성 중요도 확인
m_dt.feature_importances_
df_iris.feature_names

s1 = Series(m_dt.feature_importances_, index = df_iris.feature_names)
s1.sort_values(ascending=False)


# 7. 시각화
# 1. graphviz 설치(window)
# download 후 압축해제(C:/Program Files (x86))
 
# download : https://graphviz.gitlab.io/_pages/Download/Download_windows.html


# 2. graphviz 설치(python)
# C:\Users\KITCOOP> pip install graphviz


# 3. 파이썬 graphvis path 설정
import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


# 4. 파이썬 시각화(cancer data)
df_cancer = load_breast_cancer()

train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

m_dt = dt(random_state=0)
m_dt.fit(train_x, train_y)

import graphviz

from sklearn.tree import export_graphviz
export_graphviz(m_dt,                           # 모델명 
                out_file="tree.dot", 
                class_names=df_cancer.target_names,
                feature_names=df_cancer.feature_names, 
                impurity=False, 
                filled=True)

with open("tree.dot", encoding='UTF8') as f:
    dot_graph = f.read()

g1 = graphviz.Source(dot_graph)
g1.render('a1', cleanup=True) 

########## 여기까지는 복습입니다. ##########

# RF(iris data) in python
# 1. 데이터 로딩
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2. 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state=0)

# 3. 모델 학습(32)
m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)    # n_estimators=100, n_jobs=None

# 4. 모델 평가
m_rf.score(test_x, test_y)    # 97.37

# 5. 매개변수 튜닝
v_score_te = []

for i in range(1,101) :
    m_rf = rf(random_state=0, n_estimators = i)
    m_rf.fit(train_x, train_y)
    v_score_te.append(m_rf.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(np.arange(1,101), v_score_te, color='red')    

# 6. 특성 중요도 파악
m_rf.base_estimator_
m_rf.feature_importances_



# [ 연습 문제 - cancer data의 분류 모델 생성 및 비교 ]
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

df_cancer2 = pd.read_csv('cancer.csv')

# 1. knn
m_knn = knn(3)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)    # 92

m_knn2 = knn(3)
m_knn2.fit(train_x[:, 2:4], train_y)
m_knn2.score(test_x[:, 2:4], test_y)    # 89

# 2. DT
m_dt = dt(random_state=0)
m_dt.fit(train_x, train_y)
m_dt.score(test_x, test_y)    # 88

# 3. RF
m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)
m_rf.score(test_x, test_y)    # 97


# 특성 중요도 시각화
s1 = Series(m_rf.feature_importances_, index = df_cancer.feature_names)
s1.sort_values(ascending=False)

def plot_feature_importances_cancer(model, data) : 
    n_features = data.data.shape[1]  # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    

plt.rc('font',family='Malgun Gothic')    
plot_feature_importances_cancer(m_rf, df_cancer)


# 트리기반 모델
# DT > RF > GB > XGB > ...

C:\Users\KITCOOP> pip install xgboost

from xgboost.sklearn import XGBClassifier as xgb
from xgboost.sklearn import XGBRegressor as xgb_r


# 분석 시 고려 사항
# 1. 변수 선택
# 2. 변수 변형(결합 포함) ***
# 3. 교호작용(interaction)
# 4. 교차검증(cross validation)
# 5. 최적의 매개변수 조합(grid search)
# 6. 변수 표준화(scaling)


# 교호 작용(interaction)
# - 변수 상호간 서로 결합된 형태로 의미 있는 경우
# - 2차, 3차항... 추가 가능
# - 발생 가능한 모든 다차항의 interaction으로 부터 의미 있는 변수 추출
