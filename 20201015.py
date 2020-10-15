# 트리기반 모델
# DT > RF > GB > XGB > ...

C:\Users\KITCOOP> pip install xgboost

from xgboost.sklearn import XGBClassifier as xgb
from xgboost.sklearn import XGBRegressor as xgb_r

# Gradient Boosting Tree(GB)
# - 이전 트리의 오차를 보완하는 트리를 생성하는 구조
# - 비교적 단순한 초기 트리를 형성, 오분류 data point에 더 높은 가중치를 부여,
#   오분류 data point를 정분류 하도록 더 보안된, 복잡한 트리를 생성
# - learning rate 만큼의 오차 보완률 결정(높을수록 과적합 발생 가능성)
# - random forest 모델보다 더 적은수의 tree로도 높은 예측력을 기대할 수 있음
# - 각 트리는 서로 독립적이지 않으므로 병렬처리에 대한 효과를 크게 기대하기 어려움

# GB(iris data) in python
# 1. 데이터 로딩 및 분리
run profile1
from sklearn.datasets import load_iris
df_iris = load_iris()

train_x, test_x, train_y, test_y = train_test_split(df_iris.data, 
                                                    df_iris.target, 
                                                    random_state=99)
# 2. 모델 생성 및 학습
from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.ensemble import GradientBoostingRegressor as gb_r

m_gb = gb()
m_gb.fit(train_x, train_y)  # learning_rate=0.1, max_depth=3,
                            # max_features=None, min_samples_split=2,
                            # n_estimators=100
# 3. 모델 평가
m_gb.score(test_x, test_y)  # 97.37

# 4. 매개 변수 튜닝
vscore_tr = [] ; vscore_te = []

for i in [0.001, 0.01, 0.1, 0.5, 1] :
    m_gb = gb(learning_rate = i)
    m_gb.fit(train_x, train_y)
    vscore_tr.append(m_gb.score(train_x, train_y))
    vscore_te.append(m_gb.score(test_x, test_y))

plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_tr, label = 'train_score')
plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_te, label = 'test_score',
         color='red')

plt.legend()   
  
# 5. 특성 중요도 시각화
def plot_feature_importances(model, data) : 
    n_features = data.data.shape[1]  # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    
m_gb = gb(learning_rate=0.1)
m_gb.feature_importances_

plot_feature_importances(m_gb, df_iris)




# 변수 스케일링
# - 설명변수의 서로 다른 범위를 동일한 범주내 비교하기 위한 작업
# - 거리기반 모델, 회귀계수의 크기 비교, NN의 모델 등에서 필요로 하는 작업
# - 각 설명변수의 중요도를 정확히 비교하기 위해서도 요구되어짐
# - interaction 고려시 결합되는 설명변수끼리의 범위를 동일하게 만들 필요 있음

# 1. scaling 종류
# 1) standard scaling : 변수를 표준화 하는 작업(평균이 0, 표준편차 1) 
#    표준화 : (X - 평균) / 표준편차
from sklearn.preprocessing import StandardScaler as standard

# 2) minmax scaling : 최소값을 0으로, 최대값을 1로 만드는 작업
from sklearn.preprocessing import MinMaxScaler as minmax
    

# 2. scaling 실행
# 1) standard scaling
m_sc1 = standard()
m_sc1.fit(train_x)         # 각 설명변수의 평균, 표준편차 계산      
m_sc1.transform(train_x)   # 표준화 계산

m_sc1.transform(train_x).mean(axis=0)  # 0에 근사
m_sc1.transform(train_x).std(axis=0)   # 1에 수렴

# 2) minmax scaling 
m_sc2 = minmax()
m_sc2.fit(train_x)                      # 각 설명변수의 최대, 최소 구하기 

m_sc3 = minmax()
m_sc3.fit(test_x)  

train_x_sc = m_sc2.transform(train_x)   # 최소를 0, 최대를 1에 맞춰 계산
test_x_sc = m_sc2.transform(test_x)     # 최소를 0, 최대를 1에 맞춰 계산

m_sc2.transform(train_x).min(axis=0)    # 0
m_sc2.transform(train_x).max(axis=0)    # 1

m_sc2.transform(test_x).min(axis=0)    # 0이 아님
m_sc2.transform(test_x).max(axis=0)    # 1이 아님

m_sc3.transform(test_x).min(axis=0)    # 0
m_sc3.transform(test_x).max(axis=0)    # 1


# [ 연습 문제 : iris data의 knn모델 적용시 scaling 전/후 비교 ]
# 1. scaling 전
m_knn = knn(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)             # 92

# 2. scaling 후
m_knn = knn(5)
m_knn.fit(train_x_sc, train_y)
m_knn.score(test_x_sc, test_y)          # 94

# 3. 변수 선택 및 scaling 후 
m_knn = knn(5)
m_knn.fit(train_x_sc[:,2:4], train_y)
m_knn.score(test_x_sc[:,2:4], test_y)   # 97


# [ 참고 : minmax scaling 방식의 비교 ]
m_sc2 = minmax()
m_sc2.fit(train_x)                      # 각 설명변수의 최대, 최소 구하기 

m_sc2.fit_transform(train_x)            # fit과 transform 동시에 처리

m_sc3 = minmax()
m_sc3.fit(test_x)  

# 1) 올바른 scaling
train_x_sc1 = m_sc2.transform(train_x)   
test_x_sc1  = m_sc2.transform(test_x)    

# 2) 잘못된 scaling
train_x_sc2 = m_sc2.transform(train_x)   
test_x_sc2  = m_sc3.transform(test_x) 

# 시각화
# 1) figure와 subplot 생성(1X3)
fig, ax = plt.subplots(1,3)    

# 2) 원본 data(x1, x2)의 산점도
import mglearn

plt.rc('font', family='Malgun Gothic')
ax[0].scatter(train_x[:,0], train_x[:,1], c=mglearn.cm2(0), label='train')
ax[0].scatter(test_x[:,0], test_x[:,1], c=mglearn.cm2(1), label='test')
ax[0].legend()
ax[0].set_title('원본 산점도')
ax[0].set_xlabel('sepal length')
ax[0].set_ylabel('sepal width')


# 3) 올바른 scaling후 데이터(x1, x2)의 산점도
ax[1].scatter(train_x_sc1[:,0], train_x_sc1[:,1], c=mglearn.cm2(0),
              label='train')
ax[1].scatter(test_x_sc1[:,0], test_x_sc1[:,1], c=mglearn.cm2(1), 
              label='test')
ax[1].legend()
ax[1].set_title('올바른 스케일링 산점도')
ax[1].set_xlabel('sepal length')
ax[1].set_ylabel('sepal width')

# 4) 잘못된 scaling후 데이터(x1, x2)의 산점도
ax[2].scatter(train_x_sc2[:,0], train_x_sc2[:,1], c=mglearn.cm2(0),
              label='train')
ax[2].scatter(test_x_sc2[:,0], test_x_sc2[:,1], c=mglearn.cm2(1),
              label='test')    
ax[2].legend()
ax[2].set_title('잘못된 스케일링 산점도')
ax[2].set_xlabel('sepal length')
ax[2].set_ylabel('sepal width')

# => train data set과 test data set이 분리되어진 상태일 경우
#    각각 서로 다른 기준으로 scaling을 진행하면(3번째 subplot)
#    원본의 데이터와 산점도가 달라지는 즉, 원본의 데이터의 왜곡이 발생
#    따라서 같은 기준으로 train/test를 scaling 하는 것이 올바른 scaling 방식!!
   
   
# 교호 작용(interaction)
# - 변수 상호간 서로 결합된 형태로 의미 있는 경우
# - 2차, 3차항... 추가 가능
# - 발생 가능한 모든 다차항의 interaction으로 부터 의미 있는 변수 추출

# 1. interaction 적용 data 추출
from sklearn.preprocessing import PolynomialFeatures as poly

원본      => 2차항 적용(transform)
x1 x2 x3    x1^2  x2^2  x3^2  x1x2  x1x3  x2x3   
1  2   3     1     4     9      2    3     6
2  4   5     4     16    25     8    10    20


m_poly = poly(degree=2)
m_poly.fit(train_x)         # 각 설명변수에 대한 2차항 모델 생성
m_poly.transform(train_x)   # 각 설명변수에 대한 2차항 모델 생성

m_poly.get_feature_names()                       # 변경된 설명변수들의 형태 확인
m_poly.get_feature_names(df_iris.feature_names)  # 실제 컬럼이름의 교호작용 출력
 
DataFrame(m_poly.transform(train_x) , 
          columns = m_poly.get_feature_names(df_iris.feature_names))

