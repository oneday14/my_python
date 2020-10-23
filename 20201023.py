# 문자 변수 -> 숫자 변수 변경
run profile1
df1 = DataFrame({'col1':[1,2,3,4],
                 'col2':['M','M','F','M'],
                    'y':['N','Y','N','Y']})

# 1. if문으로 직접 치환
np.where(df1.col2 == 'M', 0, 1)


# 2. dummy 변수 치환 함수 사용
pd.get_dummies(df1, drop_first=True)

# 3. 기타 함수 사용
# 1)
mr = pd.read_csv("mushroom.csv", header=None)

def f_char(df) : 
    target = []
    data = []
    attr_list = []
    for row_index, row in df.iterrows() :
        target.append(row.iloc[0])             # Y값 분리
        row_data = []
        for v in row.iloc[1:] :                # 설명변수를 모두 숫자로 변환
            row_data.append(ord(v))            # ord를 사용한 숫자 변환 방식
        data.append(row_data)
    return DataFrame(data)

f_char(mr)
f_char(df1)

# 위 코드 해석 참고
# 1) ord 함수
ord('a')    # 97
ord('abc')  # length error

# 2) iterrows 메서드 역활 : 각 행을 행번호와 함께 분리
for row_index, row in mr.iterrows():
    print(str(row_index) + ':' + str(list(row)))
    
# => 7678:['e', 'b', 'f', 'g', 'f', 'n', 'f', 'w', 'b', 'g', 'e', '?', 's', 'k', 'w', 'w', 'p', 'w', 't', 'p', 'w', 's', 'g']


# LabelEncoder 방식에 의한 문자열의 숫자 변환
from sklearn.preprocessing import LabelEncoder

df2 = DataFrame({'col1':[1,2,3,4],
                 'col2':['ABC1','BCD1','BCD2','CDF1'],
                    'y':['N','N','N','Y']})

f_char(df2)   # 에러 발생 : 2자 이상의 문자열은 숫자 변경 불가

m_label = LabelEncoder()
m_label.fit(df2.col2)        # 1차원만 학습 가능
m_label.transform(df2.col2)  # 값의 unique value마다 서로 다른 숫자 부여

def f_char2(df) : 
    def f1(x) :
        m_label = LabelEncoder()
        m_label.fit(x)
        return m_label.transform(x)
    
    return df.apply(f1, axis=0)

# 보스턴 주택 데이터 가격 셋
from sklearn.datasets import load_boston
df_boston = load_boston()

df_boston.data
df_boston.feature_names
df_boston.target          # 종속변수가 연속형

print(df_boston.DESCR)

# 2차 interaction이 추가된 확장된 boston data set
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

# 회귀 분석
- 통계적 모델이므로 여러가지 통계적 가정 기반
- 평가 메트릭이 존재
- 인과 관계 파악
- 이상치에 민감
- 다중공선성(설명 변수끼리의 정보의 중복) 문제를 해결하지 않으면
  잘못된 회귀 계수가 유도될 수 있음(잘못된 유의성 결과, 반대의 부호로 추정)
- 학습된 설명변수가 많을수록 모델 자체의 설명력이 대체적으로 높아짐(overfit)  
- 기존 회귀를 최근에는 분류모델기반 회귀나 NN로 대체해서 많이 사용
- 연속형 변수의 범주화(binding)에 따라 성능이 좋아질 수 있음

ex) 연속형 변수의 범주화(binding) 예제
Y : 성적
x1 : 성별
x2 : 나이
x3 : IQ
a4x4 : 공부시간

x5
5시간 이하 : 0
5시간 이상 : 1

# 1. 회귀 모델 적용(sklearn 모델)
from sklearn.linear_model import LinearRegression

m_reg = LinearRegression()
m_reg.fit(df_boston.data, df_boston.target)
m_reg.score(df_boston.data, df_boston.target)  # R^2 : 74


# R^2
- 회귀분석에서의 모델을 평가하는 기준
- SSR/SST  : 총 분산중 회귀식으로 설명할 수 있는 분산의 비율
- 0 ~ 1의 값을 갖고, 1에 가까울수록 좋은 회귀 모델
- 대체적으로 설명변수가 증가할수록 높아지는 경향

# SST, SSR, SSE
y - ybar = (y - yhat) + (yhat - ybar) 
sum((y - ybar)^2) = sum((y - yhat)^2) + sum((yhat - ybar)^2) 
SST(총편차제곱합)   = SSE(오차제곱항)     + SSR(회귀제곱항)  
MST(총분산)        = MSE(평균제곱오차)    + MSR(회귀제곱평균)  

고정                 작아짐               커짐   좋은 회귀식일수록

# 2. 2차항 설명변수 추가 후 회귀 모델 적용
m_reg = LinearRegression()
m_reg.fit(df_boston_et_x, df_boston_et_y)
m_reg.score(df_boston_et_x, df_boston_et_y)  # R^2 : 92.9

