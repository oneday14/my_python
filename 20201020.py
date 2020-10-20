# [ 연습 문제 : grid search - random forest cancer data set ]

run profile1

# 1) data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

# 2) data split
trainval_x, test_x, trainval_y, test_y = train_test_split(df_cancer.data, 
                                                          df_cancer.target,
                                                          random_state=0)

train_x, val_x, train_y, val_y = train_test_split(trainval_x, 
                                                  trainval_y,
                                                  random_state=0)

# 3) 모델 학습 및 매개변수 튜닝(min_samples_split, max_features)
# min_samples_split : 2 ~ 10
# max_features : 1 ~ 30

# 3-1) 교차 검증 수행 X
best_score = 0
for i in range(2,11) :       # min_samples_split
    for j in range(1,31) :   # max_features
        m_rf = rf(min_samples_split=i, max_features=j, random_state=0)
        m_rf.fit(train_x, train_y)
        vscore = m_rf.score(val_x, val_y)
        
        if vscore > best_score :
            best_score = vscore
            best_params = {'min_samples_split' : i, 'max_features' : j}
            
            
m_rf = rf(**best_params, random_state=0)
m_rf.fit(trainval_x, trainval_y)
m_rf.score(test_x, test_y)

# 3-2) 교차 검증 수행(매개변수 튜닝 시)
from sklearn.model_selection import cross_val_score

best_score = 0
for i in range(2,11) :       # min_samples_split
    for j in range(1,31) :   # max_features
        m_rf = rf(min_samples_split=i, max_features=j, random_state=0)
        cr_score = cross_val_score(m_rf, trainval_x, trainval_y, cv = 5)
        
        vscore = cr_score.mean()
        if vscore > best_score :
            best_score = vscore
            best_params = {'min_samples_split' : i, 'max_features' : j}
                   
m_rf = rf(**best_params, random_state=0)
m_rf.fit(trainval_x, trainval_y)
m_rf.score(test_x, test_y)


# grid search 기법
# - 위의 중첩 for문을 사용한 매개변수의 조합을 찾는 과정을 함수화
# - CV 기법을 포함시켜 validation data set에 대한 교차 test를 수행

from sklearn.model_selection import GridSearchCV

# 1. 모델 생성
m_rf = rf()

# 2. 그리드 서치 기법을 통한 매개변수 조합 찾기
# 2-1) 매개변수 조합 생성
v_params = {'min_samples_split' : np.arange(2,11), 
            'max_features' : np.arange(1,31)}

# 2-2) 그리드 서치 모델 생성
m_grid = GridSearchCV(m_rf,        # 적용 모델
                      v_params,    # 매개변수 조합(딕셔너리)
                      cv=5)

# 2-3) 그리드 서치에 의한 모델 학습
m_grid.fit(trainval_x, trainval_y)

# 2-4) 결과 확인
dir(m_grid)
m_grid.best_score_    # 베스트 매개변수 값을 갖는 평가 점수
m_grid.best_params_   # {'max_features': 1, 'min_samples_split': 8}


df_result = DataFrame(m_grid.cv_results_)
df_result.T.iloc[:,0] # 첫 번째 매개변수 셋 결과

# 2-5) 최종 평가
m_grid.score(test_x, test_y)

# 2-6) 그리드 서치 결과 시각화
df_result.mean_test_score      # 교차 검증의 결과(5개의 점수에 대한 평균)
arr_score = np.array(df_result.mean_test_score).reshape(30, 9)

import mglearn
plt.rc('figure', figsize=(10,10))
plt.rc('font', size=6)

mglearn.tools.heatmap(arr_score,                      # 숫자 배열
                      'min_samples_split',            # x축 이름(컬럼)
                      'max_features',                 # y축 이름(행)
                      v_params['min_samples_split'],  # x축 눈금
                      v_params['max_features'],       # y축 눈금
                      cmap='viridis')

# [ 참고 : df_result의 결과 reshape 시 배치 순서 ]
#                           min_samples_split
#                          2       3          4    5        ...  10
# 'max_features': 1   0.957811 0.962490  0.964815 0.960164
# 'max_features': 2   
#                      ......
# 'max_features': 30                      


