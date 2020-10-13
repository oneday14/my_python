# 보스턴 주택 가격 데이터 셋(회귀 분석 데이터)
from sklearn.datasets import load_boston
df_boston = load_boston()
df_boston.keys()           # 'data', 'target', 'feature_names', 'DESCR'

df_boston.data.shape       # (506, 13)

# 보스턴 주택 가격 데이터 셋(2차원 interaction 추가)
import mglearn
boston_x, boston_y = mglearn.datasets.load_extended_boston()
boston_x.shape             # (506, 104)

13(1차원) + 13(2차원) + 78
13 + 13 + 78

# 파이썬 combination 출력
import itertools
list(itertools.combinations(['x1','x2','x3'],2))   # choose(10,2) in R

# 교호작용(interaction) 데이터 셋
- 기존 설명 변수 : x1, x2, x3
- 2차원 교호작용 추가 : x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2
- 3차원 교호작용 추가 : x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2, 
                      x1x2x3, x1^3, x2^3, x3^3
                      
주택가격 <- x1(지하 주차장 공급 면적) * x2(강수량)
게임승률 <- x1(kill수)/x2(게임수/게임시간)



# DT(iris data) in python
#1. data loading
from sklearn.datasets import load_iris
df_iris = load_iris()

# 1. data split
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state=0)

# 2. 모델 생성
m_dt = dt()     # 매개변수 기본값

# 3. 훈련셋 적용(모델 훈련)
m_dt.fit(train_x, train_y)  # min_samples_split=2
                            # max_depth=None
                            # max_features=None

# 4. 모델 평가
m_dt.score(test_x, test_y)  # 97.37

# 5. 매개 변수 튜닝
- min_samples_split : 각 노드별 오분류 개수 기반으로 추가 split을 할 지 결정
                      min_samples_split 값보다 오분류 개수가 크면 split
                      min_samples_split 값이 작을수록 모델은 복잡해지는 경향
                      
- max_features : 각 노드의 고정시킬 설명변수의 후보의 개수
                 max_features 클수록 서로 비슷한 트리로 구성될 확률 높아짐
                 max_features 작을수록 서로 다른 트리 구성, 
                 복잡한 트리를 구성할 확률 높아짐
                 
- max_depth : 설명변수의 중복 사용의 최대 개수
              max_depth 작을수록 단순한 트리를 구성할 확률이 높아짐
              
from sklearn.model_selection import cross_val_score
m_dt = dt(random_state=0)
v_score = cross_val_score(m_dt,             # 적용 모델 
                          df_iris.data,     # 전체 설명변수 데이터 셋
                          df_iris.target,   # 전체 종속변수 데이터 셋
                          cv=5)             # 교차검증 횟수

v_score.mean()


