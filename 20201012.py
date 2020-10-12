# 데이터 분석
# 데이터로부터 의미 있는 정보를 추출, 정보를 통해 미래를 예측하는 모든 행위

# ex) 
# - 주가, 날씨, 수요 예측
# - 양/악성 유무 예측
# - 고객 이탈 예측
# - 연관성 분석(장바구니 분석)
# - 고객의 세분화


# [ 데이터 분석의 분류 ]
# 1. 지도 학습 : Y값(target)을 알고 있는 경우의 분석
# 1) 회귀 분석 : Y가 연속형
# 2) 분류 분석 : Y가 이산형(범주형)

# 2. 비지도 학습 : Y값(target)을 알고 있지 않은 경우의 분석
# 1) 군집 분석 : 데이터 축소 테크닉(세분류)
# 2) 연관성 분석 : 장바구니 분석

# ex) 목적 : 종양의 양/악성 유무 판별 예측

# f(x) = y 
# y : 양성, 악성
# x : 종양의 크기, 모양, 색깔, ...

# 파이썬에서의 데이터 분석 환경
# 1. numpy : 숫자 데이터의 빠른 연산 처리
# 2. pandas : 정형 데이터의 입력
# 3. scikit-learn : 머신러닝 데이터 분석을 위한 모듈(샘플 데이터 제공, 알고리즘)
#                   anaconda에 포함된 모듈
# 4. scipy : 조금 더 복잡한 과학적 연산 처리 가능(선형대수..)
#            anaconda에 포함된 모듈
# 5. matplotlib : 시각화
# 6. mglearn : 외부 모듈, 분석에서의 복잡한 시각화 함수 제공
# C:\Users\KITCOOP> pip install mglearn
# C:\Users\KITCOOP> ipython
# In [1]: import mglearn

# 데이터 분석 과정
# 1. 분석 목적(따릉이 고장 예측)
# 2. 데이터 수집
# 3. 변수 선택 및 가공
# 4. 모델 학습 및 예측
# 5. 튜닝
# 6. 비지니스 모델 적용

# scikit-learn에서의 sample data 형식
from sklearn.datasets import load_iris
iris_dataset = load_iris()   # 딕셔너리 구조로 저장

iris_dataset.keys()

iris_dataset.data            # 설명변수 데이터(array 형식)
iris_dataset.feature_names   # 각 설명변수 이름
iris_dataset.target          # 종속변수 데이터(array 형식)
iris_dataset.target_names    # Y의 각 값의 이름(factor level name)
print(iris_dataset.DESCR)    # 데이터 설명

# [ 참고 : Y값의 학습 형태 ]
# ex) 이탈예측 : 이탈, 비이탈의 범주를 갖는 종속변수
# Y : 이탈, 비이탈
# Y : 0, 1
# Y_0, Y_1 : 두개 종속변수로 분리

# Y      Y   Y_0(이탈)    Y_1(비이탈)
# 이탈    0      1            0
# 이탈    0      1            0
# 비이탈  1      0            1


# knn
# - 분류 모델(지도학습)
# - 거리기반 모델
# - input data와 가장 거리가 가까운 k개의 관측치를 통해 input data의 Y값 결정
# - 이상치에 민감한 모델
# - 스케일링에 매우 민감(변수의 스케일 표준화)
# - 학습되는 설병변수의 조합에 매우 민감
# - 내부 feature selection 기능 없음


# knn(iris data) in python

# 1. 데이터 분리(train/test)
X = iris_dataset.data
Y = iris_dataset.target

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,              # X data
                                                    Y,              # Y data
                                                    train_size=0.7, # train set 추출 비율
                                                    random_state=99) # seed값 고정
# =============================================================================
# [ 참고 : row number 추출 방식(R 방식) ]
# import random
# iris_nrow = iris_dataset.data.shape[0]
# rn = random.sample(range(iris_nrow), round(iris_nrow*0.7))
# 
# iris_train = iris_dataset.data[rn, ]
# iris_test = DataFrame(iris_dataset.data).drop(rn, axis=0).values
# =============================================================================

# [ 참고 - sample 함수 ]
random.sample([1,20,31,49,10],  # 추출 대상
              1)                # 추출 개수(정수)

# 2. 데이터 학습
from sklearn.neighbors import KNeighborsClassifier as knn_c 
from sklearn.neighbors import KNeighborsRegressor as knn_r

m_knn = knn_c(5)
m_knn.fit(train_x, train_y)
m_knn.predict(test_x)

# 3. 평가
# sum(m_knn.predict(test_x) == test_y) / test_x.shape[0] * 100
m_knn.score(test_x, test_y) * 100

# 4. 예측
new_data = np.array([6.1, 2.9, 5.3, 1.9])
new_data1 = np.array([[6.1, 2.9, 5.3, 1.9]])

m_knn.predict(new_data.reshape(1,-1))
m_knn.predict(new_data1)

iris_dataset.target_names[2]
iris_dataset.target_names[m_knn.predict(new_data1)[0]]

# 5. 튜닝
# k수 변화에 따른 train, test score 시각화
score_tr = [] ; score_te = []

for i in range(1,11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x, train_y)
    score_tr.append(m_knn.score(train_x, train_y))
    score_te.append(m_knn.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(range(1,11), score_tr, label = 'train_score')
plt.plot(range(1,11), score_te, label = 'test_score', color = 'red')
plt.legend()

# 6. 데이터 시각화
import mglearn
df_iris = DataFrame(X, columns= iris_dataset.feature_names)
pd.plotting.scatter_matrix(df_iris,               # X data set
                           c=Y,                   # Y data set(color표현) 
                           cmap = mglearn.cm3,    # color mapping 
                           marker='o',            # 산점도 점 모양
                           s=60,                  # 점 크기
                           hist_kwds={'bins':30}) # 히스토그램 인자 전달




















