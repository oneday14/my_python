# 분류 모델
# - 거리기반 모델(knn)
# - 트리기반 모델(DT, RF, GB, XGB..)
# - SVM

# SVM(Support Vector Machine)
# - 분류 기준을 회귀분석처럼 선형선, 혹은 초평면(다차원)을 통해 찾는 과정
# - 다차원(고차원) 데이터셋에 주로 적용
# - 초평면을 만드는 과정이 매우 복잡, 해석 불가(black box 모델)
# - c(비선형성 강화), gamma(고차원성 강화)의 매개변수의 조합이 매우 중요
# - 계수를 추정하는 방식이 회귀와 유사, 학습 전 변수의 scaling 필요
# - 이상치에 민감
# - 초기 분류기준으로부터 support vector에 가중치를 부여, 분류기준 강화 하는 과정

# SVM cancer data in python
from sklearn.svm import SVC

# 1. data loading
run profile1
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

# 2. data 분리
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

# 3. 모델 생성 및 학습
m_svm = SVC()
m_svm.fit(train_x, train_y)   # C=1.0, gamma='scale'

# 4. 모델 평가
m_svm.score(test_x, test_y)   # 93.71

# 5. 변수 스케일링 후 적용
from sklearn.preprocessing import StandardScaler as standard
m_sc1 = standard()

m_sc1.fit(train_x)
train_x_sc = m_sc1.transform(train_x)
test_x_sc = m_sc1.transform(test_x)

m_svm2 = SVC()
m_svm2.fit(train_x_sc, train_y)  # C=1.0, gamma='scale'
m_svm2.score(test_x_sc, test_y)  # 96.5


# 6. 매개변수 튜닝
v_parameter = {'C':[0.001,0.01,0.1,1,10,100,1000],
               'gamma':[0.001,0.01,0.1,1,10,100,1000]}

m_svm = SVC()
m_grid = GridSearchCV(m_svm, v_parameter, cv=5)
m_grid.fit(train_x_sc, train_y)
m_grid.score(test_x_sc, test_y)  # 97.9
m_grid.best_params_              # {'C': 10, 'gamma': 0.01}
df_result = DataFrame(m_grid.cv_results_)

df_result.loc[:, ['params', 'mean_test_score']]

arr_score = np.array(df_result.mean_test_score).reshape(7,7)

import mglearn
mglearn.tools.heatmap(arr_score,
                      'gamma',
                      'C',
                      v_parameter['gamma'],
                      v_parameter['C'],
                      cmap='viridis')


# overfit 확인
v_para_c = [0.001,0.01,0.1,1,10,100,1000]
v_para_gamma = [0.001,0.01,0.1,1,10,100,1000]

v_score_tr = [] ; v_score_te = [] ; v_c = [] ; v_gamma = []

for i in v_para_c :
    for j in v_para_gamma :
        m_svc = SVC(C=i, gamma=j)
        m_svc.fit(train_x_sc, train_y)
        v_score_tr.append(m_svc.score(train_x_sc, train_y))
        v_score_te.append(m_svc.score(test_x_sc, test_y))
        v_c.append(i)
        v_gamma.append(j)
        

df_result2 = DataFrame({'C':v_c,
                        'gamma':v_gamma,
                        'train_score':v_score_tr,
                        'test_score':v_score_te})

df_result2.loc[:,['train_score','test_score']].plot()

# x축 눈금 표현('C':0.001 'gamma':0.001)
f1 = lambda x, y : 'C:' + str(x) + ', ' + 'gamma:' + str(y) 

df_result2.C.map(f1, df_result2.gamma)          # 불가
l1 = list(map(f1, df_result2.C, df_result2.gamma))   # map 함수로 두 객체 전달

f1 = lambda x, y : 'C:' + str(x) + ', ' + 'gamma:' + str(y) 

df_result2.apply(f1, axis=1)   # y를 알 수 없어 불가

f2 = lambda x : 'C:' + str(x[0]) + ', ' + 'gamma:' + str(x[1]) 
df_result2.apply(f2, axis=1)   # 가능

plt.xticks(df_result2.index,   # x축 눈금(숫자)
           l1,                 # 각 눈금의 이름(문자)
           rotation=270,       # 축 이름 출력 방향
           fontsize=6)


# PCA(Principal Component Analysis : 주성분 분석)
# - 비지도학습
# - 기존의 변수로 새로운 인공변수를 유도하는 방식(변수 결합)
# - 유도된 인공변수끼리 서로 독립적
# - 첫번째 유도된 인공변수가 기존 데이터의 분산을 가장 많이 설명하는 형식
# - 회귀의 다중공선성 문제 해결
# - 기존 데이터를 모두 사용, 저차원 모델 생성 가능 => 과대적합 해소
# - 의미 있는 인공변수 유도
# - 변수 scaling 필요

Y = X1 + X2 + X3 + X4 + X5
C1 = a1X1 + a2X2 + a3X3 + a4X4 + a5X5 
C2 = b1X1 + b2X2 + b3X3 + b4X4 + b5X5

Y = c1C1 + c1C2  # PCA + regressor

# PCA + knn iris data 
# 1. data loading
from sklearn.datasets import load_iris
df_iris = load_iris()

train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state=0)

# 2. 인공변수 유도
from sklearn.decomposition import PCA
m_pca = PCA(n_components = 2)
m_pca.fit(train_x)
train_x_pca = m_pca.transform(train_x)
test_x_pca = m_pca.transform(test_x)

# 3. 인공변수 확인
dir(m_pca)
m_pca.components_     # 유도된 인공변수의 계수

# array([[ 0.37649644, -0.06637905,  0.85134571,  0.35924188],
#        [ 0.6240207 ,  0.75538031, -0.18479376, -0.07648543]])

# C1 = 0.37649644*X1 -0.06637905*X2 + 0.85134571*X3 + 0.35924188*X4
# C2 = 0.6240207*X1  +0.75538031*X2 -0.18479376*X3 -0.07648543*X4

# 4. 유도된 인공변수 knn 모델에 적용
m_knn = knn(5)
m_knn.fit(train_x_pca, train_y)
m_knn.score(test_x_pca, test_y)   # 97.36

# 5. data point들의 분포 확인(산점도)
import mglearn
mglearn.discrete_scatter(train_x_pca[:,0], train_x_pca[:,1], train_y)


# [ 연습 문제 - cancer data의 PCA + svm 적용 ]
# 1. data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

# 2. 변수 스케일링
m_stand = standard()
m_stand.fit(train_x)
train_x_sc = m_stand.transform(train_x)
test_x_sc  = m_stand.transform(test_x)

# 3. 인공변수 유도
m_pca1 = PCA(2)
m_pca2 = PCA(3)

m_pca1.fit(train_x_sc)
m_pca2.fit(train_x_sc)

train_x_sc_pca1 = m_pca1.transform(train_x_sc)
test_x_sc_pca1 = m_pca1.transform(test_x_sc)

train_x_sc_pca2 = m_pca2.transform(train_x_sc)
test_x_sc_pca2 = m_pca2.transform(test_x_sc)

# 4. SVM 모델 적용
v_parameter = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

m_svm = SVC()
m_grid = GridSearchCV(m_svm, v_parameter, cv=5)

m_grid.fit(train_x_sc_pca1, train_y)
m_grid.score(test_x_sc_pca1, test_y)   # 92.30

m_grid.fit(train_x_sc_pca2, train_y)
m_grid.score(test_x_sc_pca2, test_y)   # 92.30

# 5. 시각화
# 1) 2개 인공변수(2차원)
import mglearn
mglearn.discrete_scatter(train_x_sc_pca1[:,0],  # X축
                         train_x_sc_pca1[:,1],  # Y축
                         train_y)               # target 값에 따른 색 표현

# 2) 3개 인공변수(3차원)
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure)
    
ax.scatter(train_x_sc_pca2[train_y == 0 , 0],   # X축 좌표
           train_x_sc_pca2[train_y == 0, 1],   # Y축 좌표
           train_x_sc_pca2[train_y == 0, 2],   # Z축 좌표
           c='b',                      # color(blue)
           cmap=mglearn.cm2,           # color mapping
           s=60,                       # 점 사이즈
           edgecolor='k')              # 점 테두리 색(black)

ax.scatter(train_x_sc_pca2[train_y == 1, 0], 
           train_x_sc_pca2[train_y == 1, 1], 
           train_x_sc_pca2[train_y == 1, 2], 
           c='r', 
           cmap=mglearn.cm2, 
           s=60, 
           edgecolor='k')

# PCA + knn 
# - 초기 이미지 인식 모델
