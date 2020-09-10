# -*- coding: utf-8 -*-
run profile1

# 1.delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine='python')

# 1) 각 시군구별 업종 비율 출력
#        족발/보쌈 중국음식    치킨
# 강남구   31         45      21.5 ....
deli2 =  deli.pivot_table(index='시군구', columns='업종',values='통화건수',
                          aggfunc='sum')

f1 = lambda x : x / x.sum() * 100
deli2.apply(f1, axis=1)

# 2) 각 업종별 통화건수가 많은 순서대로 시군구의 순위를 출력
# ste1) 첫번째 컬럼에 대해 순위 부여
idx1 = deli2.iloc[:,0].sort_values(ascending=False).index
nrow1 = deli2.shape[0]
Series(np.arange(1,nrow1+1), index = idx1)

# step2) 함수 생성
def f2(x) :
    idx1 = x.sort_values(ascending=False).index
    nrow1 = deli2.shape[0]
    s1 = Series(np.arange(1,nrow1+1), index = idx1)
    return s1

# step3) 각 컬럼별 적용
deli2.apply(f2, axis=0)

# apply에 의해 매 반복마다 리턴 객체가 스칼라이면 최종 리턴은 Series
# apply에 의해 매 반복마다 리턴 객체가 Series이면 최종 리턴은 DataFrame

# [ 참고 : rank에 의한 풀이(rank는 뒤에 정리) ]
deli2.rank(ascending=False, axis=0)

# 3) top(data,n=5) 함수 생성, 업종별 통화건수가 많은 top 5 시간대 출력
# step1) 필요한 교차 테이블 생성
deli3 = deli.pivot_table(index='시간대', columns='업종', values='통화건수',
                         aggfunc='sum')

# step2) 첫번째 컬럼에 대해 수행
deli3.iloc[:,0].sort_values(ascending=False)[:5].index

# step3) 함수 생성 및 적용
top = lambda x, n=5 : x.sort_values(ascending=False)[:n].index
deli3.apply(top, n=3, axis=0)    # 업종이 컬럼일 경우 DataFrame 리턴

deli3.T.apply(top, n=3, axis=1)  # 업종이 인덱스일 경우 Series 리턴

f4 = lambda x, n=5 : Series(x.sort_values(ascending=False)[:n].index)
deli3.T.apply(f4, n=3, axis=1)  # f4 함수의 리턴을 Series로 만든 후 적용
                                # DataFrame 리턴

# 2. 부동산_매매지수현황.csv파일을 읽고
test2 = pd.read_csv('부동산_매매지수현황.csv', engine='python', skiprows=1)

# step1) multi-column 설정
idx2 = test2.columns.map(lambda x : NA if 'Unnamed' in x else x[:2])
c1 = Series(idx2).fillna(method='ffill')
c2 = test2.iloc[0,:]

test2.columns = [c1,c2]
test2 = test2.iloc[2:,:]

test2.columns.names = ['지역','구분']

# step2) 첫번째 컬럼(날짜) index 설정
test2.set_index(NA)            # NA컬럼이 multi-column 이므로
                               # index가 두 level값을 갖는 tuple로 전달

test2.index = test2.iloc[:,0]  # 첫번째 컬럼 색인 결과는 Series 이므로
                               # index가 1차원 형식으로 전달

test2 = test2.iloc[:,1:]
test2.index.name = '날짜'


# 1) 각 월별 지역별 활발함과 한산함지수의 평균을 각각 출력
# step1) 월별 그룹핑을 위한 년,월,일 3 level의 multi-index 생성
vyear = test2.index.map(lambda x : x[:4])
vmonth = test2.index.map(lambda x : x[5:7])
vday = test2.index.map(lambda x : x[8:])

test2.index = [vyear, vmonth, vday]

# step2) 숫자 컬럼으로 변경
test2 = test2.astype('float')

# step3) 월별 평균
test3 = test2.mean(axis=0, level=1)

# 2) 지역별 활발함지수가 가장 높은 년도 출력
test4 = test2.xs('활발함', axis=1, level=1).mean(axis=0, level=0)

test4.idxmax(axis=0)
test4.apply(top, n=3, axis=0)

########## 여기까지는 복습입니다. ##########

# 적용함수의 활용
# - map 함수 : 추가 인자 전달 가능(객체)
# - map 메서드 : 추가 인자 전달 불가
# - apply : 추가 키워드 인자 전달 가능(n=3)
# - applymap : 추가 인자 전달 불가

# [ 연습문제 - 적용함수 ]
# 1. emp.csv 파일을 읽고
# 사용자가 입력한 증가율이 적용된 연봉 출력(10입력시 10% 증가)
# 1) map 메서드 사용(불가)
f_sal1 = lambda x, y : round(x * (1 + y/100))
emp['SAL'].map(f_sal1, y=10)  # y인자 전달 불가

# 2) map 함수 사용(키워드 인자 전달 불가, 객체 전달 가능)
list(map(f_sal1, emp['SAL'], y=10))              # 불가
list(map(f_sal1, emp['SAL'], [10]))              # 하나만 출력
list(map(f_sal1, emp['SAL'], np.repeat(10,14)))  # 전체 출력

# 3) apply 사용(키워드 인자 전달 가능)
emp['SAL'].apply(f_sal1, y=10)                       # Series에 전달 가능
                                                     # new feature

f_sal2 = lambda x, y : round(x['SAL'] * (1 + y/100)) # ****
emp.apply(f_sal2, y=15, axis=1)

# 2. (SAL + COMM) * 증가율이 적용된 연봉 출력
# 단, 증가율은 10번 부서일 경우 10%, 20번은 11%, 30번은 12%(정수출력)
# 단, COMM이 없는 직원은 100부여
# 1) map 함수
def f_sal3(x,y,z) :   # sal, comm, deptno순 입력
    if z == 10 :
        vrate = 1.1
    elif z == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(y).fillna(100)    
    return round((x + vcomm) * vrate)

f_sal3(800,NA,10)      # Series 리턴
f_sal3(800,NA,10)[0]   # scalar 리턴

list(map(f_sal3, emp.SAL, emp.COMM, emp.DEPTNO))

# -- 함수 수정
def f_sal4(x,y,z) :   # sal, comm, deptno순 입력
    if z == 10 :
        vrate = 1.1
    elif z == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(y).fillna(100)    
    return round((x + vcomm) * vrate)[0]

list(map(f_sal4, emp.SAL, emp.COMM, emp.DEPTNO))
    
# 2) apply
emp['SAL'].apply(f_sal4, emp.COMM, emp.DEPTNO) # 객체 전달 불가

def f_sal5(x) :   # sal, comm, deptno순 입력
    if x['DEPTNO'] == 10 :
        vrate = 1.1
    elif x['DEPTNO'] == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(x['COMM']).fillna(100)    
    return round((x['SAL'] + vcomm) * vrate)[0]

emp.apply(f_sal5, axis=1)

# rank 메서드
# - 순위 출력 함수
# - R과 비슷
# - pandas 제공
# - axis 옵션 가능 : 자체 행별, 열별 적용 가능

s1 = Series([10,2,5,1,6])
s2 = Series([10,2,5,1,1,6])

s1.rank(axis,              # 진행 방향
        method={'average', # 서로 같은 순위 부여, 평균값으로
                'min',     # 서로 같은 순위 부여, 순위중 최소값으로
                'max',     # 서로 같은 순위 부여, 순위중 최대값으로
                'first'},  # 서로 다른 순위 부여, 앞에 있는 관측치에 더 높은순위
        ascending)         # 정렬 순서
s1
s1.rank()
s2
s2.rank(method='first')

# 순위        1 2 3 4 5 6
s3 = Series([1,2,2,2,3,4])
s3.rank()
s3.rank(method='min')

# DataFrame의 rank 사용
df1 = DataFrame({'col1':[4,1,3,5], 'col2':[1,2,3,4]})

df1.rank(axis=0)  # 세로방향, 같은 컬럼 내 순위 부여
df1.rank(axis=1)  # 가로방향, 같은 행 내 순위 부여

# merge 
# - 두 데이터의 join
# - 세개 이상의 데이터의 join 불가
# - equi join만 가능
# - outer join 가능

pd.merge(left,              # 첫번째 데이터 셋
         right,             # 두번째 데이터 셋
         how={'inner',      # inner join 수행(조인조건에 맞는 데이터만 출력)
              'left',       # left outer join
              'right',      # right outer join
              'outer'},     # full outer join
         on,                # join column
         left_on,           # left data join column
         right_on,          # right data join column
         left_index=False,  # left data index join 여부
         right_index=False, # right data index join 여부
         sort=False)        # 출력결과 정렬 여부

# 1) 컬럼으로 inner join
df2 = DataFrame({'col1':['a','b','c'],
                 'col2':[1,2,3]})

df3 = DataFrame({'col1':['c','b','a'],
                 'col2':[30,20,10]})

pd.merge(df2, df3, on='col1', suffixes=('_df2', '_df3'))
pd.merge(df2, df3, left_on='col1', 
                   right_on='col1', suffixes=('_df2', '_df3'))


# 2) index로 inner join
df22 = df2.set_index('col1')
df33 = df3.set_index('col1')

pd.merge(df22,df33,on='col1')  # index의 이름이 있는 경우 가능

pd.merge(df22,df33,left_index=True, 
                   right_index=True)  # index의 이름이 있는 경우 가능

# 3) outer join
df4 = DataFrame({'col1':['a','b','c','d'],
                 'col2':[1,2,3,4]})
pd.merge(df3, df4, on='col1')                # inner join 수행
pd.merge(df3, df4, on='col1', how='right')   # inner join 수행

# [ 연습 문제 ]
emp = pd.read_csv('emp.csv')
gogak = pd.read_csv('gogak.csv', engine='python')
gift = pd.read_csv('gift.csv', engine='python')

# 1. emp.csv 파일을 읽고 각 직원의 이름, 연봉, 상위관리자의 이름, 연봉 출력
emp2 = pd.merge(emp, emp, left_on='MGR', right_on='EMPNO',
                suffixes=['_직원','_관리자'],
                how='left')

emp2.loc[:,['ENAME_직원','ENAME_관리자','SAL_직원','SAL_관리자']]

# 2. gogak.csv 파일과 gift.csv 파일을 읽고
# 각 고객이 받는 상품이름을 고객이름과 함께 출력
gogak
gift.loc[(gift['G_START'] <= 980000) & (980000 <= gift['G_END']),'GNAME']
gift.loc[(gift['G_START'] <= 73000) & (73000 <= gift['G_END']),'GNAME']
gift.loc[(gift['G_START'] <= 320000) & (320000 <= gift['G_END']),'GNAME']

def f_gift(x) :
    vbool = (gift['G_START'] <= x) & (x <= gift['G_END'])
    gname = gift.loc[vbool, 'GNAME']
    return gname

gogak['POINT'].map(f_gift)        # 각 반복마다 Series형식으로 return
f_gift(gogak['POINT'][0]).iloc[0] # 리턴결과에서 원소만 가져오기 위한 색인

def f_gift2(x) :
    vbool = (gift['G_START'] <= x) & (x <= gift['G_END'])
    gname = gift.loc[vbool, 'GNAME']
    return gname.iloc[0]

gogak['POINT'].map(f_gift2)




# 날짜 변환
from datetime import datetime

# 1. strptime # str(string) p(parsing) time 
# - 문자 -> 날짜
# - datetime 모듈 호출시 가능
# - 벡터 연산 불가
# - parsing format 생략 불가

d1 = '2020/09/10'
d1.strptime()       # 문자열에 전달(메서드 형식) 불가

datetime.strptime(d1)             # 에러, 2번째 인자 필요
datetime.strptime(d1, '%Y/%m/%d') # 2번째 인자 전달 시 파싱 가능

l1 = ['2020/09/10','2020/09/11','2020/09/12']
datetime.strptime(l1, '%Y/%m/%d')  # 벡터 연산 불가

Series(l1).map(lambda x : datetime.strptime(x, '%Y/%m/%d'))

# 2. strftime # str(string) f(format) time 
# - 날짜 -> 문자(날짜의 형식 변경)
# - 메서드, 함수 형식 모두 가능
# - 벡터 연산 불가
t1 = datetime.strptime(d1, '%Y/%m/%d')
t2 = Series(l1).map(lambda x : datetime.strptime(x, '%Y/%m/%d'))

t1.strftime('%A')            # datetime object 적용 가능(메서드 형식)
datetime.strftime(t1, '%A')  # 함수 적용 가능
datetime.strftime(t2, '%A')  # 벡터 연산 불가

# [ 연습 문제 ]
# 1. 업종별 콜수가 가장 많은 요일 출력
deli = pd.read_csv('delivery.csv', engine='python')

# step1) 요일 추출
deli['일자'].strptime('%Y%m%d')  # Series 객체 전달 불가
datetime.strptime(deli['일자'], '%Y%m%d')  # parsing 대상 리스트 전달 불가

d1 = deli['일자'].map(lambda x : datetime.strptime(str(x), '%Y%m%d'))
d1.strftime('%A')                # 벡터 연산 불가

deli['요일'] = d1.map(lambda x : x.strftime('%A'))


# step2) 교차 테이블 생성
deli_idx = deli.pivot_table(index='요일', columns='업종', values='통화건수',
                            aggfunc='sum')

# step3) 업종별 콜수 많은 요일 출력
deli_idx.idxmax(axis=0)


# [ 연습 문제 ]
# movie_ex1.csv 파일을 읽고 요일별 이용비율이 가장 높은 연령대 출력

movie = pd.read_csv('movie_ex1.csv', engine='python')

# step1) 분리된 년,월,일 결합
# 1) 문자열 결합(+)의 벡터연산 활용
date1 = movie['년'].astype('str') + '/' + movie['월'].astype('str') + '/' + movie['일'].astype('str')

# 2) 적용함수 활용
f_date1 = lambda x,y,z : str(x) + '/' + str(y) + '/' + str(z)
list(map(f_date1, movie['년'], movie['월'], movie['일']))

f_date2 = lambda x : str(x['년']) + '/' + str(x['월']) + '/' + str(x['일'])
movie.apply(f_date2, axis=1)

# step2) 날짜 parsing
date2 = date1.map(lambda x : datetime.strptime(x, '%Y/%m/%d'))

# step3) 날짜 포맷 변경(요일 추출)
movie['요일'] = date2.map(lambda x : x.strftime('%A'))

# step4) 교차 테이블 생성
movie_1 = movie.pivot_table(index='요일', columns='연령대', 
                            values='이용_비율(%)', aggfunc='sum')

# step5) idx 사용
movie_1.idxmax(axis=1)


# 기본 문자열 메서드(벡터 연산 X)
# pandas 문자열 메서드(벡터 연산 O)
