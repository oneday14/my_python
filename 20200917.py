# 파이썬 날짜 표현
# 날짜 형식 : datetime

from datetime import datetime
dir(datetime)

# 1. 현재 날짜 출력
d1 = datetime.now()    # datetime.datetime(2020, 9, 17, 9, 19, 10, 613903)

d1.year   # 날짜에서 년 추출
d1.month  # 날짜에서 월 추출
d1.day    # 날짜에서 일 추출
d1.hour   # 날짜에서 시 추출
d1.minute # 날짜에서 분 추출
d1.second # 날짜에서 초 추출

# 2. 날짜 파싱(문자 -> 날짜)
# 2-1) datetime.strptime
# - 벡터 연산 불가
# - 두 번째 인자(날짜 포맷) 생략 불가
datetime.strptime('2020/09/17', '%Y/%m/%d')

# 2-2) pd.to_datetime
# - 벡터 연산 가능
# - 날짜 포맷 생략 가능(전달시에는 format='')
l1 = ['2020/01/01', '2020/09/17']
d2 = pd.to_datetime(l1)
pd.to_datetime(l1, format='%Y/%m/%d')  # format 인자 이름 생략 불가

# 2-3) datetime 함수
# - 년,월,일,시,분,초 순서대로 값을 전달하여 날짜를 생성(파싱)
# - 벡터 연산 불가

vyear = [2007,2008]
vmonth = [7,8]
vday = [7,8]

datetime(2007,9,11)          # 각 인자에 년,월,일 순서로 정수 전달
datetime(vyear,vmonth,vday)  # 각 인자에 리스트 전달 불가, 벡터연산 불가

# 3. 포맷 변경
datetime.strftime(d1,'%A')
datetime.strftime(d2,'%A')  # d2의 전달 불가

d1.strftime('%A')
d2.strftime('%A')           # d2(datetimeindex) 전달 가능, 벡터 연산 가능

# 4. 날짜 연산
t1 = datetime(2020,9,17)
t2 = datetime(2020,9,10)

t1 - t2             # 날짜 - 날짜 연산 가능(기본 단위 : 일), timedelta object
(t1 - t2).days      # timedelta object의 일 수 출력
(t1 - t2).seconds   # timedelta object의 초 수 출력 


# 1) timedelta를 사용한 날짜 연산
from datetime import timedelta

d1 + 100              # 날짜와 숫자 연산 불가
d1 + timedelta(100)   # 100일 뒤


# 2) offset으로 사용한 날짜 연산
import pandas.tseries.offsets
dir(pandas.tseries.offsets)

from pandas.tseries.offsets import Day, Hour, Second

Day(5)    # 5일
Hour(5)   # 5시간
Second(5) # 5초

d1 + Day(100)

# [ 연습 문제 ]
# emp.csv 파일을 읽고
emp = pd.read_csv('emp.csv')

# 1) 년,월,일 각각 추출
emp.HIREDATE.map(lambda x : datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
emp['HIREDATE'] = pd.to_datetime(emp.HIREDATE)

emp.HIREDATE.year     # Series의 날짜에서는 year 전달 불가
emp.HIREDATE[0].year  # scalar의 날짜에서는 year 전달 가능

emp.HIREDATE.map(lambda x : x.year)
emp.HIREDATE.map(lambda x : x.month)
emp.HIREDATE.map(lambda x : x.day)


# 2) 급여 검토일의 요일 출력 (단, 금여 검토일은 입사날짜의 100일 후 날짜)
emp.HIREDATE + 100
(emp.HIREDATE + Day(100)).strftime('%A')  # Series 객체 전달 불가
(emp.HIREDATE + Day(100)).map(lambda x : x.strftime('%A'))

# 3) 입사일로부터의 근무일수 출력
d1 - emp.HIREDATE # timedelta 객체는 X일 X초로 구분해서 출력
                  # days, seconds라는 메서드로 각각 선택하여 출력 가능

(d1 - emp.HIREDATE).days    # Series 객체 전달 불가(벡터 연산 불가)
(d1 - emp.HIREDATE)[0].days # scalar 객체 전달 가능

(d1 - emp.HIREDATE).map(lambda x : x.days)



# 5. 날짜 인덱스 생성 및 색인

# pd.date_range : 연속적 날짜 출력
pd.date_range(start,    # 시작 날짜
              end,      # 끝 날짜
              periods,  # 기간(출력 개수)
              freq)     # 날짜 빈도(매월, 매주...)

pd.date_range(start='2020/01/01', end='2020/01/31') # 기본 freq='D'(일)
pd.date_range(start='2020/01/01', periods=100) # 시작값으로부터 100일의 날짜

pd.date_range(start='2020/01/01', 
              end='2020/01/31', freq='7D') # by값과 비슷

# [ 참고 - freq의 전달 ]
pd.date_range(start='2020/01/01', end='2020/01/31', freq='D')     # 일
pd.date_range(start='2020/01/01', end='2020/01/31', freq='7D')    # 7일
pd.date_range(start='2020/01/01', end='2020/01/31', freq='W')     # 매주 일
pd.date_range(start='2020/01/01', end='2020/01/31', freq='W-WED') # 매주 수
pd.date_range(start='2020/01/01', end='2020/01/31', freq='W-MON') # 매주 월
pd.date_range(start='2020/01/01', end='2020/01/31', freq='W-MON') # 매주 월
pd.date_range(start='2020/01/01', end='2020/12/31', 
              freq='MS') # 매월 1일
pd.date_range(start='2020/01/01', end='2020/12/31', 
              freq='M') # MonthEnd, 매월 말
pd.date_range(start='2020/01/01', end='2020/12/31', 
              freq='BMS') # BusinessMonthBegin, 매월 영업일 첫 날
pd.date_range(start='2020/01/01', end='2020/12/31', 
              freq='WOM-3FRI') # WeekOfMonth 매월 셋째주 금요일


# 날짜 인덱스를 갖는 Series 생성
date1 = pd.date_range('2020/01/01','2020/03/31')
s1 = Series(np.arange(1,len(date1)+1), index=date1)

# 날짜 인덱스의 색인
s1['2020']     # 날짜에서 특정 년에 대한 색인 가능
s1['2020-03']  # 날짜에서 특정 년/월에 대한 색인 가능

# truncate : 날짜 인덱스를 갖는 경우의 날짜 선택 메서드
s1.truncate(after='2020-03-23')  # 처음 ~ 2020-03-23 까지 출력
s1.truncate(before='2020-03-23')  # 2020-03-23 ~ 끝 까지 출력

# 날짜 슬라이스
s1['2020-03-23':'2020-03-27']  # 끝 범위 포함

# [ 연습 문제 ]
# movie_ex1.csv 파일을 읽고 
# 1) 지역별(지역-시도) 요일별 영화 이용비율의 평균을 구하세요.
movie = pd.read_csv('movie_ex1.csv', engine='python')

f_datetime = lambda x,y,z : datetime(x, y, z).strftime('%A')

movie['요일'] = list(map(f_datetime, movie['년'], movie['월'], movie['일']))

movie.groupby(['지역-시도','요일'])['이용_비율(%)'].sum().unstack()



# 6. resample : 날짜의 빈도수 변경
# - 서로 다른 offset을 갖는 Series나 DataFrame의 연산시
#   offset을 맞춰놓고 연산 시 사용
# - 같은 offset끼리 그룹연산 가능(downsampling 경우)
# - upsampling : 더 많은 날짜수로 변경(주별 -> 일별)
# - downsampling : 더 적은 날짜수로 변경(일별 -> 월별)

# 1) downsampling 예제 : 일별 데이터를 주별 데이터로 변경
s1.resample(rule,    # 날짜빈도
            axis=0)  # 방향(0: index의 resample, 1: column의 resample)

s1.resample('M', how='sum')  # downsampling의 경우 how로 그룹함수를 전달,
                             # 구버전에서 가능

s1.resample('M').sum()       # downsampling의 경우 그룹함수를 추가 전달
                             # M은 MonthEnd를 의미하므로 매월 마지막 날짜 리턴
                             # 월별 grouping 기능을 가지고 있음


# 1) upsampling 예제 : 주별 데이터를 일별 데이터로 변경
date2 = pd.date_range('2020/01/01', '2020/03/31', freq='7D')
d3 = Series(np.arange(10, len(date2)*10 + 1, 10), index = date2)

d3.resample('D')                       # upsampling 경우 자동으로
                                       # 새로 생긴 날짜 생성 X
 
d3.resample('D', fill_method='ffill')  # fill_method로 이전 날짜 값
                                       # 사용 X

d3.resample('D').sum()                 # 새로 생긴 값(NA)을 0으로 치환
d3.resample('D').asfreq()              # 새로 생긴 값을 NA로 리턴

d3.resample('D').asfreq().fillna(0)    # 새로 생긴 값(NA)을 0으로 치환
d3.resample('D').asfreq().fillna(method='ffill') # fill_method='ffill'

d3.resample('D').ffill()                         # fill_method='ffill'

# multi-index를 갖는 경우 resample 전달
df3 = d3.reset_index()

df3.columns = ['date','cnt']
df3.index = [['A','A','A','A','A','A','B','B','B','B','B','B','B'],
             df3.date]

df3.resample('D').asfreq()            # MultiIndex의 레벨 전달 필요
df3.resample('D', level='date').sum() # MultiIndex의 레벨 전달 시 필요
df3.resample('D', on='date').sum()    # 날짜컬럼 지정시 필요

# 참고
# asfreq는 resample의 level 혹은 on 인자 사용시 전달 불가
# multi index의 특정 레벨로 resample 하는 경우 다른 레벨 값 생략됌


# dropna
# - NA값을 갖는 행(기본), 컬럼 제거(axis로 선택 가능)
# - NA가 하나라도 포함된 행, 컬럼 제거(조절 가능)

df_na = DataFrame(np.arange(1,26).reshape(5,5))

df_na.iloc[1,0] = NA
df_na.iloc[2,[0,1]] = NA
df_na.iloc[3,[0,1,2]] = NA
df_na.iloc[4,[0,1,2,3,4]] = NA

df_na.dropna()          # axis=0, NA가 하나라도 포함된 행을 모두 제거
df_na.dropna(how='any') # NA가 하나라도 포함된 행을 모두 제거

df_na.dropna(how='all') # 전체가 NA인 경우만 삭제
df_na.dropna(thresh=1)  # NA가 아닌 값이 1개 이상을 남기고 나머지 제외
df_na.dropna(thresh=3)  # NA가 아닌 값이 3개 이상을 남기고 나머지 제외

df_na.dropna(axis=1, thresh=3) # 컬럼 삭제, NA가 아닌 값이 3개 이상 리턴

# [ 참고 - 사용자 정의 함수 생성, NA 개수 기반 삭제 ]
pd.isnull(df_na.iloc[4,:]).sum() == 5

f_dropna = lambda x, n=1 : pd.isnull(x).sum() >= n
df_na.loc[~df_na.apply(f_dropna, axis=1, n=3), :]

# [ 연습 문제 ]
# 부동산_매매지수.csv 파일을 읽고
test1 = pd.read_csv('부동산_매매지수.csv', encoding='cp949',
                    skiprows=[0,2])

# NA 제거
test1 = test1.dropna(how='all')

# 1) 2008년 4월 7일부터 관찰된 매 주, 각 구별 매매지수 데이터로 표현
date3 = pd.date_range('2008/04/07', freq='7D', periods = test1.shape[0])

test1.index = date3

# 2) 2017년의 작년(2016년) 대비 매매지수 상승률 상위 10개 구를 상승률과 함께 출력
# sol1) 각 년도 추출 후 연산
v2007 = test1['2017'].mean(0)
v2006 = test1['2016'].mean(0)

vrate = (v2007 - v2006) / v2006 * 100
vrate.sort_values(ascending=False)[:10]

# sol2) 전체 년도의 전년도 대비 매매지수 상승률 계산 후 2017년 선택
test2 = test1.resample('Y').mean()
test3 = ((test2 - test2.shift(1)) / test2.shift(1) * 100)['2017'].T
test3.sort_values(by='2017-12-31', ascending=False)[:10]










