# -*- coding: utf-8 -*-
run profile1

# 1. multi_index_ex2 파일을 읽고
test1 = pd.read_csv('multi_index_ex2.csv', engine='python')

# 1) 멀티 인덱스, 컬럼 설정
# index 설정
c1 = test1.iloc[:,0].map(lambda x : str(x)[0])
c2 = test1.iloc[:,0].map(lambda x : str(x)[2])

test1.index = [c1,c2]
test1 = test1.iloc[:,1:]
test1.index.names = ['월','지점']

# column 설정
test1.columns

'서울'.isalpha()
', '.isalpha()
' .1'.isalpha()

v1 = test1.columns.map(lambda x : x if x.isalpha() else NA)
v2 = Series(v1).fillna(method='ffill')

test1.columns = [v2, test1.iloc[0,:]]

test1 = test1.iloc[1:,:]

test1.columns.names = ['지역','요일']

# 2) 결측치 0으로 수정
test1 = test1.fillna(0)

test1.replace('.',0).replace('-',0).replace('?',0)

test1 = test1.replace(['.',',','?','-'],0) # 여러 old 값을 동일한 값으로 치환 가능
'.,?-'.replace(['.',',','?','-'],'0')      # 문자열 replace 메서드는 불가

test1 = test1.astype('int')

# 3) 각 지점별로 요일별 판매량 총합을 출력
test1.sum(axis=0, level=1).sum(axis=1, level=1)

# 4) 각 월별 판매량이 가장 높은 지역이름 출력
test1.sum(axis=0, level=0).idxmax(axis=1)

f1 = lambda x : x.idxmax()[0]
test1.sum(axis=0, level=0).apply(f1, axis=1)

test1.sum(axis=0, level=0).sum(axis=1, level=0).idxmax(axis=1) # 정답

# 2. 병원현황.csv 파일을 읽고 
test2 = pd.read_csv('병원현황.csv', engine='python', skiprows=1)

# 1) 다음과 같은 데이터프레임으로 만들어라
#                   2013               
#                 1 2 3 4
# 신경과 강남구
#       강동구
#        ....

# 불필요한 데이터 제외
test2 = test2.loc[test2['표시과목'] != '계', :]
test2 = test2.drop(['항목','단위'], axis=1)

# index 생성
test2 = test2.set_index(['시군구명칭','표시과목'])

# column 생성
'2013. 4/4'[:4] # 년도 추출
'2013. 4/4'[6]  # 분기 추출

year = test2.columns.map(lambda x : x[:4])
qt   = test2.columns.map(lambda x : x[6])

test2.columns = [year, qt]
test2.dtypes

# level 치환
test2.sort_index(axis=0, level=1)
test2.sort_index(axis=0, level=1).swaplevel(0,1, axis=0)

# 2) 성형외과의 각 구별 총 합을 출력
# sol1) 시군구명칭, 표시과목이 index가 아닌 상황(일반 컬럼)
test3 = test2.reset_index()       # index로 설정된 값이 다시 컬럼으로 위치

test3 = test3.loc[test3['표시과목'] == '성형외과', :]
test3 = test3.drop('표시과목', axis=1)
test3.set_index('시군구명칭').sum(1)

# sol2) 멀티 인덱스인 상황
test2.xs('성형외과', axis=0, level=1).sum(1)
test2.loc[test2.index.get_level_values(1) == '성형외과', :].sum(1)


# 3) 강남구의 각 표시과목별 총 합 출력
test4 = test2.reset_index()
test4 = test4.loc[test4['시군구명칭'] == '강남구', :]
test4 = test4.drop('시군구명칭', axis=1)
test4.set_index('표시과목').sum(1)

# 4) 년도별 총 병원수의 합 출력
test2.sum(axis=1, level=0)

########## 여기까지는 복습입니다. ##########

# stack과 unstack
# - stack : wide -> long(tidy data)
# - unstack : long -> wide(cross table)

# multi-index의 stack과 unstack

# R에서의 stack과 unstack : 컬럼 단위
# python에서의 stack과 unstack : index(column 포함) 단위

# 1. Series에서의 unstack : index의 값을 컬럼화(DataFrame 리턴)
s1 = Series([1,2,3,4], index=[['A','A','B','B'],['a','b','a','b']])

s1.unstack()        # index의 가장 하위 level이 unstack 처리
s1.unstack(level=0) # 지정한 index의 level이 unstack 처리

# 2. DataFrame에서의 stack : 싱글컬럼의 값을 index화(Series 리턴) 
#                           멀티컬럼의 값을 index화(DataFrame 리턴)

df1 = s1.unstack() 
df1.stack()         # 하위 컬럼의 값이 stack 처리
df1.stack(level=0)  # 지정한 column의 level이 stack 처리

df2 = DataFrame(np.arange(1,9).reshape(2,4),
                columns=[['col1','col1','col2','col2'],
                         ['A','B','A','B']])

df2.stack()
df2.stack(level=0)

# 3. DataFrame에서의 unstack : 특정 레벨의 index를 컬럼화
df3 = df2.stack()
df3.unstack()
df3.unstack(level=0)

# [ 연습 문제 ]
# sales2.csv 파일을 읽고 
sales = pd.read_csv('sales2.csv', engine='python')
sales = sales.set_index(['날짜','지점','품목'])

# 1) 다음과 같은 형태로 만들어라
#                 냉장고          tv             세탁기         에어컨
#                 출고 판매 반품  출고 판매 반품  출고 판매 반품  출고 판매 반품
# 2018-01-01  c1 
sales_1 = sales.unstack().sort_index(axis=1, level=1).swaplevel(0,1,axis=1)

sales.stack().unstack(level=[2,3])

# 2) 위의 데이터 프레임에서 아래와 같은 현황표로 출력(총합)
# 출고  ---
# 판매  ---
# 반품  ---
sales_1.sum(axis=1, level=1).sum(0)


# [ 정리 : multi-index 색인 ] 
# 1. iloc 
# - 특정 위치 값 선택 가능

# 2. loc
# - 상위레벨의 색인 가능
# - 하위레벨의 값 선택 불가
# - 인덱스 값의 순차적 전달을 통한 색인 가능

# 예) df3에서 'A' 선택
df3.loc[[(0,'A'),(1,'A')],:]

# 3. xs
# - 하위레벨 색인 가능
# - 인덱스의 순차적 전달 없이도 바로 하위 레벨 색인 가능
# - 특정 레벨을 선택한 전달 가능(중간 레벨 스킵 가능)

# 예) df3에서 'A' 선택
df3.xs('A', axis=0, level=1)

# 예) 3개 레벨을 갖는 아래 df33에서 index가 'A' 이면서 '1'인 행 선택
df33 = DataFrame(np.arange(1,17).reshape(8,2),
                 index=[['A','A','A','A','B','B','B','B'],
                        ['a','a','b','b','a','a','b','b'],
                        ['1','2','1','2','1','2','1','2']])

df33.loc[('A','1'),:]    # 중간 레벨 생략 불가
df33.loc[('A',:,'1'),:]  # 중간 레벨 생략 불가
df33.loc[[('A','a','1'),('A','b','1')],:]  # 중간 레벨 생략 불가

df33.xs(('A','1'), level=[0,2])

# 4. get_level_values
# - index object의 method
# - 선택하고자 하는 레벨 이름 혹은 위치갑 전달
# - index의 특정 레벨 선택 후 조건 전달 방식

# 예) df33에서 세번째 레벨의 값이 '1'인 행 선택
df33.xs('1', axis=0, level=2)

df33.loc[df33.index.get_level_values(2) == '1', :]


# [ 연습 문제 ]
# movie_ex1.csv 파일을 읽고
movie = pd.read_csv('movie_ex1.csv', engine='python')

# 1) 지역-시도별, 성별 이용비율의 평균을 정리한 교차테이블 생성
movie2 = movie.set_index(['지역-시도','성별'])['이용_비율(%)']

movie2.unstack()                   # index값의 중복으로 인해 처리 불가
movie2.sum(level=[0,1]).unstack()

movie.pivot_table(index='지역-시도', columns='성별', values='이용_비율(%)',
                  aggfunc='sum')

# 2) 일별- 연령대별 이용비율의 평균을 정리한 교차테이블 생성
movie3 = movie.set_index(['일','연령대'])['이용_비율(%)']
movie3.sum(level=[0,1]).unstack()

movie.pivot_table(index='일', columns='연령대', values='이용_비율(%)',
                  aggfunc='sum')

# 3) 년~ 성별까지를 모두 인덱스로 생성, 10일 이전 데이터 선택
a1 = list(movie.columns.values[:-1])
movie4 = movie.set_index(a1)

movie.loc[movie['일'] < 10, :]
movie.loc[movie4.index.get_level_values(2) < 10]

# [ 참고 - multi column일때 sort_values로 정렬할 컬럼 전달 방법 ]
df1.sort_values(by='a', ascending=False)
df2.sort_values(by=[('col1','A'),('col2','A')], ascending=False)


# cross-table
# - wide data
# - 행별, 열별 정리된 표 형식 => 행별, 열별 연산 용이
# - join 불가
# - group by 연산 불가
# - 시각화시 주로 사용
# - multi-index를 갖는 구조를 unstack 처리하여 얻거나 pivot 통해 가능

# 예)     
# 부서     A  B  C    
# 성별   
# 남      90 89 91
# 여      89 78 95

# 1. pivot
# - 각 컬럼의 값을 교차테이블 구성요소로 전달, 교차테이블 완성
# - index, columns, values 컬럼 각각 전달
# - grouping 기능 없음(agg func)
# - index, columns 리스트 전달 불가
# - values 리스트 전달 가능

# 2. pivot_table
# - 교차 테이블 생성 메서드
# - values, index, columns 컬럼 각각 전달 (순서유의)
# - 결합기능(affregate function 전달 가능(default : mean))
# - values, index, columns컬럼에 리스트 전달 가능

# [ 예제 : 아래 데이터 프레임을 각각 교차 테이블 형태로 정리 ]
pv1 = pd.read_csv('dcast_ex1.csv', engine='python')
pv2 = pd.read_csv('dcast_ex2.csv', engine='python')
pv3 = pd.read_csv('dcast_ex3.csv', engine='python')

# 1) pv1에서 품목별 price, qty 정보를 정리한 교차표
pv1.pivot(index='name', columns='info', values='value')
pv1.set_index(['name','info'])['value'].unstack()

# 2) pv2에서 년도별, 품목별 판매현황 정리
pv2.pivot('year','name',['qty', 'price'])

# 3) pv3에서 년도별 음료의 판매현황(수량) 정리
pv3.pivot('년도','이름','수량')           # 중복값이 있어 불가
pv3.pivot(['년도','지점'],'이름','수량')   # 리스트 전달 불가

pv3.pivot_table(index=['년도','지점'],    # 리스트 전달 가능
                columns='이름',
                values='수량')

pv3.pivot_table('수량',['년도','지점'],'이름')  # 인자 이름 생략시
                                              # values, index, columns순

pv3.pivot_table('수량','년도','이름')                 # 요약기능 가능
pv3.pivot_table('수량','년도','이름', aggfunc='sum')  # 요약함수 전달 가능



# [ 연습문제 ]
# movie_ex1.csv 파일을 읽고
movie = pd.read_csv('movie_ex1.csv', engine = 'python')

#1) 지역-시도별, 성별 이용 비율의 평균을 정리한 교차테이블 생성
movie.pivot_table('이용_비율(%)', '지역-시도', '성별', aggfunc = 'sum' )

#2) 일별-연령대별 이용 비율의 평균을 정리한 교차테이블 생성
movie.pivot_table('이용_비율(%)', '일', '연령대', aggfunc = 'sum' )

# delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine = 'python')

# 1. 시간대별 배달콜수가 가장 많은 업종 1개 출력
deli.pivot_table(index = '업종', columns = '시간대', values = '통화건수', aggfunc = 'sum')

d1 = deli.pivot_table(index = '시간대', columns = '업종', values = '통화건수', aggfunc = 'sum')
d1.idxmax(axis = 1)
