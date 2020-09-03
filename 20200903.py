# numpy - array 구조
# pandas - Series(단 하나의 데이터 타입만 허용), DataFrame 구조

# 적용함수
# 1. map 함수
# - 1차원 자료에 각 원소별 함수의 적용
# - 리스트로의 출력만 가능
# - 추가 인자(데이터 셋) 전달 가능

# ex) list(map(f1,l1)), list(map(f1,l1,l2))

# 2. map 메서드
# - 1차원 자료에 각 원소별 함수의 적용
# - pandas 제공
# - Series로 리턴
# - 추가 인자(데이터 셋) 전달 불가

# 3. apply 메서드
# - 2차원 자료에 각 행별, 컬럼별 함수의 적용
# - 적용함수는 그룹함수의 형태(다수의 인자를 전달받아 하나를 리턴)
 
# 4. applymap 메서드
# - 2차원 자료에 각 원소별 적용(2차원 형태 유지)
# - DataFrame으로 리턴

run profile1

# 1. test3 파일을 불러온 후 소수점 둘째자리로 표현
pd.read_csv('test3.txt', sep='\t', header=None)  # DataFrame 구조
# read.csv('test3.txt', sep='\t', header=F)  -- in R

arr1 = np.loadtxt('test3.txt', delimiter='\t')          # array 구조

# 함수 생성
'%.2f' % 3
f1 = lambda x : '%.2f' % x

list(map(f1,arr1))           # 2차원 형식 전달 불가
list(map(f1,arr1[0]))        # 1차원 형식 전달 가능

arr1.applymap(f1)            # numpy(array)에 적용 불가

from pandas import DataFrame
DataFrame(arr1).applymap(f1) # pandas(DataFrame)에 적용 가능

# 2. read_test.csv 파일을 읽고
np.loadtxt('read_test.csv', delimiter=',', dtype='str') 
df2 = pd.read_csv('read_test.csv')

df2.dtypes

# 1) .,-,?,!,null,nan 값을 모두 0으로 수정
# sol1) 각 컬럼별 치환(np.where)
df2.a = np.where((df2.a == '.') | (df2.a == '-'), 0, df2.a)

# sol2) 각 컬럼별 치환(map)
f2 = lambda x : 0 if x in ['.', '-'] else x
list(map(f2, df2.a))
df2.a.map(f2)

# sol3) 데이터 프레임 전체 적용

# =============================================================================
# [ 참고 - python에서의 NA 체크 ]
#
# np.isnan('a')                # np.isnan은 문자열에 대한 NA 확인 불가
# np.isnan(NA)      
#
# pd.isnull('a')               # pd.isnull은 문자열에 대한 NA 확인 가능
# pd.isnull(NA) 
# pd.isnull(Series([1,2,NA]))  # 벡터연산 가능
# =============================================================================

def f3(x) :
    if (x in ['.','-','?','!']) : 
        return 0
    else :
        return x

df2 = df2.applymap(f3)        

def f4(x) :
    if np.isnan(float(x)) : 
        return 0
    else :
        return x

df2 = df2.applymap(f4)        

# 특수기호와 NA인 경우 동시 0으로 치환하는 경우
def f3(x) :
    if (x in ['.','-','?','!']) | np.isnan(x) : 
        return 0
    else :
        return x

df2.applymap(f3)       # 에러 발생, np.isnan 함수가 문자열 input 허용 X

pd.isnull(0)
pd.isnull('0')
pd.isnull(NA)

def f3(x) :
    if (x in ['.','-','?','!']) | pd.isnull(x) : 
        return 0
    else :
        return x

df2 = df2.applymap(f3)               # 가능, pd.isnull 함수가 문자열 input 허용

# 2) 컬럼별 총 합 출력
# step1) 데이터프레임 각 컬럼 숫자로 변경
df2 = df2.astype('int')         # 형변환 메서드(astype)는 
                                # 2차원 데이터셋 벡터 연산 가능

int(df2)                        # 형변환 함수는 벡터 연산 불가
df2.applymap(lambda x : int(x)) # applymap으로 2차원 데이터셋에 적용

df2.dtypes

# step2) 컬럼별 총 합 계산
df2.dtypes
df2.apply(sum, axis=0)

# 3) year, month, day 컬럼 생성
'20190101'[:4]
df2.date.astype('str')[:4]    # 원소별 4개의 문자열 추출 의미 X

f5 = lambda x : str(x)[:4] 
f6 = lambda x : str(x)[4:6] 
f7 = lambda x : str(x)[6:8] 
       
df2.year = df2.date.map(f5)
df2.month = df2.date.map(f6)
df2.day = df2.date.map(f7)

# 4) d값이 가장 높은 날짜 확인
df2.date[np.argmax(df2.d)]

# 3. crime.csv 파일을 읽고
df3 = pd.read_csv('crime.csv', encoding='cp949')

# 1) 검거/발생*100 값을 계산 후 rate 컬럼에 추가
df3.rate = df3.검거 / df3.발생 * 100

# =============================================================================
# 데이터프레임의 컬럼 생성 방법
# df1.column_name = 값
# =============================================================================

# pd.read_csv 에서 파일 불러올때 NA 처리할 문자열 전달 방법
df2 = pd.read_csv('read_test.csv')        # a,b 컬럼이 문자열 컬럼
df3 = pd.read_csv('read_test.csv', na_values=['?','.','-','!'])
# 'nan', 'null' 문자열은 자동으로 nan처리

df3.dtypes

########## 여기까지는 복습입니다. ##########

# 실습1(과제 제출용)

# 1. emp.csv 파일을 읽고(dataframe으로 불러오기)
df1 = pd.read_csv('emp.csv')

# 1) A로 시작하는 직원의 연봉 출력
# df2[조건결과 , :]    # 데이터프레임 색인 불가
# df2.컬럼1[조건결과]   #  시리즈 색인 가능

'ALLEN'[0] == 'A'
'ALLEN'.startswith('A')

df1.ENAME.map(lambda x : x[0] == 'A')
df1.ENAME.map(lambda x : x.startswith('A'))       # 가능

df1.ENAME.applymap(lambda x : x.startswith('A'))  # 불가

df1.SAL[df1.ENAME.map(lambda x : x[0] == 'A')]
df1.loc[df1.ENAME.map(lambda x : x[0] == 'A'), 'SAL']  # loc(색인메서드)

# 2) 입사년도 출력
df1.HIREDATE.map(lambda x : x[:4])

# 2. card_history.txt 파일을 읽고
df2 = pd.read_csv('card_history.txt', sep='\t')

# 1) 각 품목별 지출금액의 총 합 계산
# step1) 데이터프레임 전체 데이터에 천단위 구분기호 제거
df2.applymap(lambda x : int(str(x).replace(',','')))

df3 = df2.applymap(lambda x : str(x).replace(',',''))
df3 = df3.astype('int')

df3.dtypes

# step2) 컬럼별 총 합
df3.apply(sum, axis=0)[1:]

# map 함수, 메서드***
# apply 메서드
# applymap 메서드

# 형변환 함수/메서드


# pandas
# - Series, DataFrame 생성 및 연산에 필요한 함수가 내장된 모듈
# - 특히 결측치(NA)에 대한 연산이 빠르고 쉽다
# - 산술연산에 대한 빠른 벡터연산 가능(함수 및 메서드 제공)
# - 문자열 처리 메서드는 벡터연산 불가 => mapping 필요(str 메서드 사용시 가능**)

# Series
# - DataFrame을 구성하는 요소(1차원)
# - 동일한 데이터 타입만 허용(하나의 컬럼이 주로 Series 형태)
# - key(row number/name)-value 구조

# 1. 생성
s1 = Series([1,2,3,4])
s2 = Series([1,2,3,4,'5'])
s3 = Series([1,2,3,4,'5'], index=['a','b','c','d','e'])
s4 = Series([10,20,30,40], index=['a','b','c','d'])
s5 = Series([100,200,300,400], index=['a','b','c','d'])
s6 = Series([100,200,300,400], index=['A','b','c','d'])

# 2. 연산
s1 + 10   # Series와 스칼라의 산술연산 가능
s4 + s5   # 동일한 크기이면서 index값이 같은 Series끼리 연산 가능
s4 + s6   # key가 매칭되지 않는 값의 연산은 NA 리턴

s1 = Series([1,2,3,4], index=['a','b','c','d'])
s2 = Series([10,20,30], index=['c','a','b'])

s1 + s2   # 서로다른 크기를 갖는 Series라도 동일한 key가 있으면 연산

# 3. 색인
s1[0]           # 정수색인(위치색인) 가능
s1[0:1]         # 차원축소 방지를 위한 슬라이스 색인

s1[0:2]         # 슬라이스 색인 가능
s1[[0,1,2]]     # 리스트 색인 가능
s1[s1 > 3]      # 조건 색인 가능

s1['a']         # 이름 색인 가능
s1.a            # key 색인 (in R : s1$a)

# 4. 기본 메서드
s1.dtype
s1.dtypes

s1.index    # index(row number/name) 확인
s1.values   # Series를 구성하는 value(데이터) 확인

# 5. reindex : index 재배치
s1[['c','b','a','d']]                 # c,b,a,d 순 가능
Series(s1, index=['c','b','a','d'])   # c,b,a,d 순 가능
s1.reindex(['c','b','a','d'])         # c,b,a,d 순 가능

# [ 예제 ]
# 다음의 리스트를 금,화,수,월,목,일,토 인덱스값을 갖도록 시리즈로 생성 후
# 월~일 순서로 재배치
L1=[4,3,1,10,9,5,1]

s2 = Series(L1,index=['금','화','수','월','목','일','토'])

s2[['월','화','수','목','금','토','일']]
Series(s2, index = ['월','화','수','목','금','토','일'])
s2.reindex(['월','화','수','목','금','토','일'])

# DataFrame
- index(행)와 column(컬럼)으로 구성
- key(column)-value 구조

# 1. 생성
d1 = {'col1':[1,2,3,4], 'col2':['a','b','c','d']}
arr1 = np.arange(1,9).reshape(4,2)

df1 = DataFrame(d1)
df2 = DataFrame(arr1)
df3 = DataFrame(arr1, index=['A','B','C','D'], columns=['col1','col2'])

# 2. 색인
df3[0,0]      # 기본적 색인 처리 불가
df3['col1']   # 컬럼의 이름 전달 가능

df3.loc       # label indexing(이름 색인)
df3.iloc      # positional indexing(위치 색인)

df3.iloc[0,0] 
df3.iloc[[0,1],0] 
df3.iloc[[0,1],0:2] 

df3.iloc[[0,1],'col2']    # 에러, iloc 메서드에 이름은 전달 불가
df3.loc[[0,1],'col2']     # 에러, iloc 메서드에 위치값 전달 불가

df3.iloc[df3.col1 > 5, :] # boolean값 전달은 iloc 메서드로 불가
df3.loc[df3.col1 > 5, :]  # boolean값 전달은 loc 메서드로 가능

