run profile1

# 1. 'test3.txt' 파일을 읽고 
df1 = pd.read_csv('test3.txt', sep='\t', header=None)

# 1) 다음과 같은 데이터 프레임 형태로 변경
# 	     20대	30대 40대 50대 60세이상
# 2000년	  7.5	3.6	 3.5  3.3	1.5
# 2001년	  7.4	3.2	 3	  2.8	1.2
# 2002년	  6.6	2.9	 2	  2	    1.1
# ....................................
# 2011년	  7.4	3.4	 2.1  2.1	2.7
# 2012년	  7.5	3	 2.1  2.1	2.5
# 2013년	  7.9	3	 2	  1.9	1.9

# index 설정
np.arange(2000,2014) + '년'
Series(np.arange(2000,2014)).astype('str').map(lambda x : x + '년')
s1 = Series(np.arange(2000,2014)).map(lambda x : str(x) + '년')

df1.index = s1
df1.index.name = 'year'

# column 설정
s2 = Series(np.arange(20,61,10)).map(lambda x : str(x) + '대')
s2[-1]                   # -를 사용한 reverse indexing이 Series에 전달 X
s2.iloc[-1] = '60세이상'  # iloc 메서드를 사용

df1.columns = s2
df1.columns.name = '연령대'

# 2) 2010년부터의 20~40대 실업률만 추출하여 새로운 데이터프레임 생성
df1.loc['2010년':, '20대':'40대']

# 3) 30대 실업률을 추출하되, 소수점 둘째자리의 표현식으로 출력
df1['30대'].map(lambda x : '%.2f' % x)

# 4) 60세 이상 컬럼 제외
df1.loc[:, '20대':'50대']
df1.drop('60세이상', axis=1)

# 5) 30대 컬럼의 값이 높은순 정렬
df1.sort_values(by='30대', ascending=False)

# 2. subway2.csv  파일을 읽고
sub = pd.read_csv('subway2.csv', engine='python', skiprows=1)

# 1) 다음의 데이터 프레임 형식으로 변경
# 전체     구분   5시       6시     7시 ...
# 서울역  승차 17465  18434  50313 ...
# 서울역  하차 ....

# 역 이름 채우기
len(sub['전체'])     # 1차원 원소의 개수
sub.shape[0]         # 2차원 데이터의 행의 수
sub.shape[1]         # 2차원 데이터의 컬럼의 수

l1=[]

for i in range(0, len(sub['전체'])) :
    if pd.isnull(sub['전체'][i]) :
        l1.append(sub['전체'][i-1])
    else :
        l1.append(sub['전체'][i])

l2 = Series(l1).map(lambda x : x.split('(')[0])

sub['전체'] = l2

# 컬럼이름 변경
sub.columns[2:] = values... # 이런 형태 불가
a1 = sub.columns.values

str(int('05~06'[:2])) + '시'
a1[2:] = Series(a1[2:]).map(lambda x : str(int(x[:2])) + '시')

sub.columns = a1

# 2) 각 역별 하차의 총 합
sub.dtypes

# 하차 추출
sub_1 = sub.loc[sub['구분'] == '하차', :]
sub_1 = sub_1.drop('구분', axis=1)

# 역이름 컬럼을 index 설정
sub_1.index = sub_1['전체']

# 전체컬럼 제외
sub_1 = sub_1.drop('전체', axis=1)

sub_1.apply(sum, axis=1)
sub_1.sum(axis=1)

# 3) 승차의 시간대별 총 합
# 승차 추출
sub_2 = sub.loc[sub['구분'] == '승차', :]

# 구분 컬럼 제외
sub_2 = sub_2.drop('구분', axis=1)

# 전체컬럼(역이름) index 설정
sub_2 = sub_2.set_index('전체')     # index 설정 후 본문에서 해당 컬럼 즉시 제외

sub_2.apply(sum, axis=0)
sub_2.sum(axis=0)

# 4) 하차 인원의 시간대별 각 역의 차지 비율 출력
(sub_1['5시'] / sub_1['5시'].sum() * 100).sum()

f1 = lambda x : x / x.sum() * 100

sub_1.apply(f1,axis=0)

########## 여기까지는 복습입니다. ##########

# NA 치환
df1 = DataFrame({'col1':[1,NA,2,NA,3],
                 'col2':['a','b','c','d',NA]})

# 1) np.where
np.where(pd.isnull(df1.col2), 'e', df1.col2)  # 1차원(컬럼) 가능
np.where(pd.isnull(df1), 'e', df1)            # 2차원 가능

# 2) 조건 치환
df1.col2[pd.isnull(df1.col2)] = 'e'           # 직접 수정 방식
df1[pd.isnull(df1)] = 'e'                     # 2차원 직접 수정 방식 불가
df1.loc[pd.isnull(df1)] = 'e'                 # 2차원 직접 수정 방식 불가

# 3) 적용함수의 사용
df1.col2.map(lambda x : 'e' if pd.isnull(x) else x) # 1차원 map으로 가능
df1.applymap(lambda x : 'e' if pd.isnull(x) else x) # 2차원 applymap 가능

# 4) NA 치환 함수
df1.col2.fillna('e')  # 1차원(Series) 데이터셋 NA 치환 가능
df1.fillna('e')       # 2차원(DataFrame) 데이터셋 NA 치환 가능

df1.fillna({'col1':0, 'col2':'e'})  # 딕셔너리 전달로 컬럼별 서로다른 값 치환

df1.fillna(method='ffill')  # 이전 값으로의 치환
df1.fillna(method='bfill')  # 다음 값으로의 치환

# 5) pandas replace 메서드 활용(밑에 정리)
df1.replace(NA,0)

# replace 메서드
# 1. 문자열 메서드 형태(기본 파이썬 제공)
# - 문자열 치환만 가능
# - 패턴치환 가능
# - 벡터 연산 불가
# - 문자값 이외 old값 사용 불가
# - 문자값 이외 new값 사용 불가

'abcde'.replace('a','A')
'abcde'.replace('abcde','A')

1.replace(1,0)      # 에러, 숫자에 replace 호출 불가
'1'.replace(1,0)    # 에러, old값은 숫자 불가
'1'.replace('1',0)  # 에러, new값은 숫자 불가
NA.replace(NA,'0')  # 에러, NA값은 치환 불가

df1.applymap(lambda x : x.replace(NA,0)) # 불가


# 2. pandas 값 치환 메서드 형태(pandas 제공)
# - 값 치환, 패턴치환 불가
# - NA(old value) 치환 가능
# - NA로(new value) 치환 가능
# - 벡터 연산 가능

df1.replace(NA,0)   # pandas에서 제공하는 replace 메서드 호출
df1.replace(1,0)    # 숫자 치환 가능
df1.replace(1,NA)   # NA로의 치환 가능

df1.iloc[0,1] = 'abcde'
df1.replace('a','A')     # 패턴 치환 불가('a'라는 값만 치환 가능)

# 예제) 
# 아래 데이터 프레임 생성 후
df1 = DataFrame({'a':[10,NA,9,1], 
                 'b':['abd','bcd','efc','add']})

# 1. 10의 값을 100으로 수정
df1[df1 == 10] = 100            # 조건 치환 불가(R에서는 가능)
df1.loc[df1 == 10] = 100        # 조건 치환 불가(R에서는 가능)

df1.replace(10, 100)

# 2. NA값을 0으로 수정
df1.replace(NA,0)
df1.fillna(0)

# 3. 데이터프레임에 있는 알파벳 d를 D로 수정
df1.replace('d','D')  # 치환 발생 X (패턴 치환은 불가하므로)
'abcd'.replace('d','D')

df1.applymap(lambda x : str(x).replace('d','D'))
df1.b.map(lambda x : str(x).replace('d','D'))


# 산술연산의 브로드캐스팅 기능**
# 브로드캐스팅 : 서로 다른 크기의 배열, 데이터프레임이 반복 연산되는 개념
# 1. array에서의 브로드캐스팅 기능
arr1 = np.arange(1,9).reshape(4,2)

arr1 + arr1[:,0]    # (4X2) + (1X4) => 불가
arr1 + arr1[:,0:1]  # (4X2) + (4X1) => 가능

# 2. DataFrame에서의 브로드캐스팅 기능
df2 = DataFrame(arr1)

df2 + df2.iloc[:,0]    # df2의 key(0,1) + Series의 key(0,1,2,3)
df2 + df2.iloc[:,0:1]  # df2의 key(0,1) + df2.iloc[:,0:1]의 key(0) 

df2 + df2.iloc[0,:]    # df2의 key(0,1) + df2.iloc[0,:]의 key(0,1)


arr1 + arr1[:,0:1]
df2.add(df2.iloc[:,0], axis=0)
df2.add(df2.iloc[:,0:1], axis=0) # add 메서드로 브로드캐스팅 기능 구현 시
                                 # DataFrame + Series 형태여야 함

# [ 연습 문제 ]
# 1. 3 X 4 배열 생성 후 a,b,c,d 컬럼을 갖는 df1 데이터프레임 생성
a1 = np.arange(1,13).reshape(3,4)
df1 = DataFrame(a1, columns=['a','b','c','d'])

# 2. 2 X 4 배열 생성 후 a,b,c,d 컬럼을 갖는 df2 데이터프레임 생성
a2 = np.arange(1,9).reshape(2,4)
df2 = DataFrame(a2, columns=['a','b','c','d'])

# 3. 위 두 데이터프레임 union 후 df3 생성
df3 = df1.append(df2, ignore_index=True)

# 4. df3에서 0,2,4 행 선택해서 새로운 데이터 프레임 df4 생성
df4 = df3.iloc[[0,2,4],:]

# 5. df3에서 'b','d' 컬럼 선택 후 새로운 데이터 프레임 df5 선택
df5 = df3.loc[:,['b','d']]

# 6. df3 - df4 수행(NA 리턴 없이)
df3 - df4
df3 - df4.reindex(df3.index)  # NA를 포함하는 연산은 NA를 리턴
df3 - df4.reindex(df3.index).fillna(0)  # NA를 포함하는 연산은 NA를 리턴
df3.sub(df4, fill_value=0)

# 7. 다음의 데이터 프레임에서 2000년 기준 가격 상승률 출력
df1 = DataFrame({'2000':[1000,1100,1200],
                 '2001':[1150,1200,1400],
                 '2002':[1300,1250,1410]}, index = ['a','b','c'])

(1150 - 1000) / 1000 * 100                       # 스칼라 연산
(df1['2001'] - df1['2000']) / df1['2000'] * 100  # Series 연산

# 사칙연산 메서드 활용
df1.sub(df1['2000'], axis=0).div(df1['2000'], axis=0) * 100

# 행, 열 전치후 브로드캐스팅 연산(사칙연산 메서드 필요 X)
df2 = df1.T
((df2 - df2.loc['2000',:]) / df2.loc['2000',:] * 100).T


# =============================================================================
# [ 참고 - 산술연산 메서드 종류 ] 
#
# DataFrame.add(+) : Add DataFrames.
# DataFrame.sub(-) : Subtract DataFrames.
# DataFrame.mul(*) : Multiply DataFrames.
# DataFrame.div(/) : Divide DataFrames (float division).
# DataFrame.truediv : Divide DataFrames (float division).
# DataFrame.floordiv(몫) : Divide DataFrames (integer division).
# DataFrame.mod(나머지) : Calculate modulo (remainder after division).
# DataFrame.pow(지수) : Calculate exponential power.
# 
# =============================================================================



# Multi-Index
# - index가 여러 층(level)을 갖는 형태
# - 파이썬 Multi-Index 지원(R에서는 불가)
# - 각 층은 level로 선택 가능(상위레벨(0))

# 1. Multi-Index 생성
     col1  col2
A a
  b    3
B a
  b

df11 = DataFrame(np.arange(1,9).reshape(4,2))
df11.index = ['a','b','c','d']

# df11.index = [[상위레벨값], [하위레벨값]]
df11.index = [['A','A','B','B'], ['a','b','a','b']]
df11.columns = ['col1', 'col2']
df11.index.names = ['상위레벨','하위레벨']

# 예제) 다음의 데이터 프레임 생성

# 		col_a		    col_b	
#		col1	col2	col1	col2
# A	a	1	      2     3       4
#	b	5	      6	    7    	8
# B	a	9	      10	11   	12
#	b	13	      14	15   	16

df22 = DataFrame(np.arange(1,17).reshape(4,4),
                 index = [['A','A','B','B'], ['a','b','a','b']],
                 columns = [['col_a','col_a','col_b','col_b'],
                            ['col1','col2','col1','col2']])

# [ 연습 문제 ]
# multi_index.csv 파일을 읽고 멀티 인덱스를 갖는 데이터프레임 변경

df33 = pd.read_csv('multi_index.csv', engine='python')

# step1) 첫번째 컬럼(지역) NA 치환(이전값)
df33.iloc[:,0] = df33.iloc[:,0].fillna(method='ffill')

# step2) 멀티 인덱스 설정
df33.index = [df33.iloc[:,0], df3.iloc[:,1]]
df33 = df33.set_index(['Unnamed: 0','Unnamed: 1'])

# step3) 멀티 인덱스 이름 변경
df33.index.names = ['지역','지점']

# step4) 컬럼 이름 변경(냉장고, 냉장고, TV, TV)
# sol1) 직접 수정
df33.columns =['냉장고','냉장고','TV','TV']

# sol2) Unnamed를 포함한 값을 NA로 수정후 이전값 치환
'Unnamed' in 'Unnamed: 3'
col1 = df33.columns.map(lambda x : NA if 'Unnamed' in x else x )

col1.fillna(method='ffill')         # index object가 호출하는 fillna는
                                    # method 옵션 사용 불가

df33.columns = Series(col1).fillna(method='ffill') # Series로 변경 후 처리

# step5) 멀티 컬럼 설정
# [ 현재컬럼, 첫번째행]

df33.columns = [df33.columns, df33.iloc[0,:]]

# step6) 첫번째 행 제거
df33 = df33.iloc[1:,]
df33.drop(NA, axis=0, level=0)  # 멀티 인덱스일 경우 레벨 전달 필요

# step7) 멀티 컬럼 이름 부여
df33.columns.names = ['품목','구분']
