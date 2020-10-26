# 보스턴 주택가격 데이터 셋
run profile1
from sklearn.datasets import load_boston
df_boston = load_boston()

train_x, test_x, train_y, test_y = train_test_split(df_boston.data,
                                                    df_boston.target,
                                                    random_state=0)

# 2차 interaction이 추가된 확장된 boston data set
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

train_x_et, test_x_et, train_y_et, test_y_et = train_test_split(
                                                    df_boston_et_x,
                                                    df_boston_et_y,
                                                    random_state=0)

from sklearn.linear_model import LinearRegression

# 회귀분석 적용
m_reg1 = LinearRegression()
m_reg1.fit(train_x, train_y)
m_reg1.score(train_x, train_y)  # 76.98
m_reg1.score(test_x, test_y)    # 63.55

m_reg2 = LinearRegression()
m_reg2.fit(train_x_et, train_y_et)
m_reg2.score(train_x_et, train_y_et)  # 95.21
m_reg2.score(test_x_et, test_y_et)    # 60.75

m_reg1.coef_   # 회귀 계수
m_reg2.coef_   # 모델과 회귀 계수에 대한 유의성 검정의 결과 출력 X


# 회귀분석의 유의성 검정 결과 확인 : statmodels 패키지(모듈)

import statsmodels.api as sm

m_reg3 = sm.OLS(train_y, train_x).fit()

dir(m_reg3)
print(m_reg3.summary())

                        OLS Regression Results   
===========================================================================                             
Dep. Variable:                      y   R-squared (uncentered):                   0.963
Model:                            OLS   Adj. R-squared (uncentered):              0.962
Method:                 Least Squares   F-statistic:                              738.2
Date:                Mon, 26 Oct 2020   Prob (F-statistic):                   2.27e-253
Time:                        09:35:21   Log-Likelihood:                         -1122.8
No. Observations:                 379   AIC:                                      2272.
Df Residuals:                     366   BIC:                                      2323.
Df Model:                          13                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.1163      0.040     -2.944      0.003      -0.194      -0.039
x2             0.0458      0.016      2.895      0.004       0.015       0.077
x3            -0.0341      0.070     -0.485      0.628      -0.172       0.104
x4             2.5435      1.014      2.508      0.013       0.549       4.538
x5            -0.0774      3.812     -0.020      0.984      -7.573       7.419
x6             5.9751      0.346     17.252      0.000       5.294       6.656
x7            -0.0153      0.016     -0.975      0.330      -0.046       0.016
x8            -0.9255      0.222     -4.178      0.000      -1.361      -0.490
x9             0.1046      0.074      1.423      0.156      -0.040       0.249
x10           -0.0086      0.004     -2.027      0.043      -0.017      -0.000
x11           -0.4486      0.126     -3.566      0.000      -0.696      -0.201
x12            0.0141      0.003      4.608      0.000       0.008       0.020
x13           -0.3814      0.058     -6.615      0.000      -0.495      -0.268
==============================================================================
Omnibus:                      179.629   Durbin-Watson:                   2.029
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1538.969
Skew:                           1.798   Prob(JB):                         0.00
Kurtosis:                      12.194   Cond. No.                     8.70e+03
==============================================================================

# 회귀분석에서의 다중공선성 문제
# - 모형은 유의하나 회귀 계수의 유의성 문제
# - 유의하다 판단되는 회귀 계수의 유의성 문제(p-value가 큼)
# - 예상한 회귀 계수의 부호와 다른 부호로 추정되는 경우

# 다중공선성 문제 해결
# - 변수 제거(더 간단한 모델로 변경)
# - 변수 결합(정보의 중복이 있는 변수끼리 결합)
# - PCA에 의한 전체 변수 결합
# - 기타 모델 적용(릿지, 라쏘)

# 릿지 회귀
# - 의미가 약한 변수의 회귀 계수를 0에 가깝게 만듬
# - 변수를 축소함으로 모델이 갖는 정보의 중복을 줄임
# - 다중공선성의 문제를 어느 정도 개선할 수 있음
# - alpha라는 매개변수 튜닝을 통해 모델의 복잡도를 제어할 수 있음
from sklearn.linear_model import Ridge

m_ridge = Ridge()
m_ridge.fit(train_x_et, train_y_et)
m_ridge.score(train_x_et, train_y_et)  # 88.58
m_ridge.score(test_x_et, test_y_et)    # 75.28

train_x_et.shape[1]             # 설명변수의 개수(104)
sum(abs(m_reg2.coef_) > 0.1)    # 104개 설명변수의 의미있는 학습
sum(abs(m_ridge.coef_) > 0.1)   # 회귀계수의 절대값이 0.1보다 큰 설명변수 수

# 매개 변수 튜닝
v_alpha = [0.001, 0.01, 0.1, 1, 10, 100]

v_score_tr = [] ; v_score_te = []

for i in v_alpha :
    m_ridge = Ridge(alpha = i)
    m_ridge.fit(train_x_et, train_y_et)
    v_score_tr.append(m_ridge.score(train_x_et, train_y_et))
    v_score_te.append(m_ridge.score(test_x_et, test_y_et))

plt.plot(v_score_tr, label='train_score')
plt.plot(v_score_te, label='test_score', c='red')
plt.legend()
plt.xticks(np.arange(len(v_alpha)), v_alpha)

# 라쏘 회귀
# - 의미가 약한 변수의 회귀 계수를 0으로 만듬 => 변수 제거 효과
# - 변수를 축소함으로 모델이 갖는 정보의 중복을 줄임
# - 다중공선성의 문제를 어느 정도 개선할 수 있음
# - alpha라는 매개변수 튜닝을 통해 모델의 복잡도를 제어할 수 있음
from sklearn.linear_model import Lasso

m_lasso = Lasso()
m_lasso.fit(train_x_et, train_y_et)
m_lasso.score(train_x_et, train_y_et)  # 29.32
m_lasso.score(test_x_et, test_y_et)    # 20.94

sum(m_lasso.coef_ == 0)                

m_lasso2 = Lasso(alpha=10)
m_lasso2.fit(train_x_et, train_y_et)
m_lasso2.score(train_x_et, train_y_et)  # 29.32
m_lasso2.score(test_x_et, test_y_et)    # 20.94

sum(m_lasso2.coef_ == 0)                # 104


# 매개변수 튜닝
v_alpha = [0.001, 0.01, 0.1, 1, 10, 100]

v_score_tr = [] ; v_score_te = []

for i in v_alpha :
    m_lasso = Lasso(alpha = i)
    m_lasso.fit(train_x_et, train_y_et)
    v_score_tr.append(m_lasso.score(train_x_et, train_y_et))
    v_score_te.append(m_lasso.score(test_x_et, test_y_et))

plt.plot(v_score_tr, label='train_score')
plt.plot(v_score_te, label='test_score', c='red')
plt.legend()
plt.xticks(np.arange(len(v_alpha)), v_alpha)

m_rfr = rf_r()
m_rfr.fit(train_x_et, train_y_et)
m_rfr.score(train_x_et, train_y_et)  # 98.76
m_rfr.score(test_x_et, test_y_et)    # 77
