run profile1

# 1. card_history.csv파일을 읽고
card = pd.read_csv('card_history.csv', engine='python')

# 1) 2018년 1월 1일부터 매주 일요일에 기록된 자료 가정, 인덱스 생성
d1 = pd.date_range('2018/01/01', periods= card.shape[0], freq='W-SUN')
card.index = d1
card = card.drop('NUM', axis=1)

# 2) 월별 각 항목의 지출 비율 출력
card = card.applymap(lambda x : int(x.replace(',','')))
card2 = card.resample('M').sum()

f1 = lambda x : round(x / x.sum() * 100,2)
card2.apply(f1, axis=1)

# 3) 일별 데이터로 변경하고, 각 일별 지출내용은 하루 평균 지출값으로 나타낸다
# 예) 1월 7일 14000원이면 1월 1일~ 1월 7일 각 2000원씩 기록
card.resample('D').asfreq()  # 1월 1일~ 1월 6일 출력 X

d2 = pd.date_range('2018/01/01', '2018/07/29')
card3 = card.reindex(d2)
card3.fillna(method='bfill') / 7


# 2. 병원현황.csv 파일을 읽고
test2 = pd.read_csv('병원현황.csv', engine='python', skiprows=1)

# 불필요 컬럼 제거
test2 = test2.drop(['항목','단위'], axis=1)

# 계 데이터 제외
test2 = test2.loc[test2['표시과목'] != '계', :]

# multi-index 생성
test2 = test2.set_index(['시군구명칭','표시과목'])
test2

# multi-column 생성
c1 = test2.columns.str[:4]
c2 = test2.columns.str[6]

test2.columns = [c1,c2]

# 1) 구별 년도별 각 표시과목(진료과목)의 이전년도 대비 증가율 출력
# (단, 각 데이터는 누적데이터로 가정)
# 4분기 시점의 데이터 추출
test3 = test2.xs('4', axis=1, level=1)

test3.shift(-1, axis=1)  # 2013년 컬럼 NA 리턴
                         # shift로 값을 이동해도 원래 컬럼의 데이터타입 유지
                         # 2013년 컬럼은 원래 정수, 2012년 실수 값을 input 시도

test3 = test3.astype('float')
test3 = test3.fillna(0)
test4 = test3.shift(-1, axis=1)

((test3 - test4) / test4 * 100).fillna(0)

# 2) 구별 년도별 병원이 생성된 수를 기반,
# 구별 년도별 가장 많이 생긴 표시과목을 각 구별로 5개씩 병원수와 함께 출력
test5 = test3 - test4
test6 = test5.stack().groupby(level=[0,2,1]).sum()

f_sort = lambda x : x.sort_values(ascending=False)[:5]

test6.groupby(level=[0,1], group_keys=False).apply(f_sort)

########## 여기까지는 복습입니다. ##########

# 파이썬 시각화

# figure와 subplot
# - figure : 그래프가 그려질 전체 창(도화지 개념)
# - subplot : 실제 그림이 그려질 공간(분할 영역)
# - in R : par(mfrow = c(1,3)) => 하나의 도화지에 3개의 분할 영역
# - figure와 subplot에 이름 부여 가능
# - 기본적으로 하나의 figure와 하나의 subplot이 생성


# 1. figure와 subplot 생성
run profile1
import matplotlib.pyplot as plt

# 1) figure와 subplot 각각 생성
fig1 = plt.figure()         # figure 생성
ax1 = fig1.add_subplot(2,   # figure 분할 행의 수
                       2,   # figure 분할 컬럼의 수
                       1)   # 분할된 subplot의 위치(1부터 시작)

ax2 = fig1.add_subplot(2,   # figure 분할 행의 수
                       2,   # figure 분할 컬럼의 수
                       2)   # 분할된 subplot의 위치(1부터 시작)

s1 = Series([1,10,2,25,4,3])
ax2.plot(s1)                 # ax2 subplot에 직접 plot 도표 전달
                             # python에서는 subplot의 위치를 직접 지정 가능

# 2) figure와 subplot 동시 생성
# - plt.subplots로 하나의 figure와 여러 개의 subplot 동시 생성
# - 이름 부여시 figure 이름과 subplot 대표 이름 각각 지정
# - subplot 위치 지정은 색인

plt.subplots(nrows=1,      # figure 분할 행의 수
             ncols=1,      # figure 분할 컬럼의 수
             sharex=False, # 분할된 subplot의 x축 공유 여부
             sharey=False) # 분할된 subplot의 x축 공유 여부


fig2, ax = plt.subplots(2,2)
ax[0,1].plot(s1)

# [ 참고 - 각 창에서 시각화모드(pylab) 전환 방법 ]
# 1. anaconda prompt(ipython)

# 1) 일반모드로 아나콘다 전환 후 pylab 모드 변경
# C:\Users\KITCOOP> ipython   # cmd에서 실행
# In [1]: %matplotlib qt      # anaconda prompt에서 실행

# 2) pylab 아나콘다 모드 직접 전환
# C:\Users\KITCOOP> ipython --pylab  # cmd에서 실행
 
# 2. spyder tool 
# Tools > Preferences > Ipython console > Graphics > Graphics backend에서
# Bacend를 Automatic으로 변경 후 spyder restart


# 2. 선 그래프 그리기
# 1) Series 전달
ax[0,0].plot(s1)     # 특정 figure, subplot에 그리는 방법
s1.plot()            # 가장 마지막 figure 혹은 subplot에 전달, 새로 생성
plt.plot(s1)

# 2) DataFrame 전달
# - 컬럼별 서로 다른 선 그래프 출력
# - 컬럼 이름 값으로 자동으로 범례 생성(위치 가장 좋은 자리)
# - index의 이름이 자동으로 x축 이름 전달
# - column의 이름이 자동으로 범례 이름 전달
 

# [ 예제 - 선그래프 그리기 ]
# fruits.csv 파일을 읽고 과일별 판매량 증감 추이 시각화
fruits = pd.read_clipboard()
fruits2 = fruits.pivot('year', 'name', 'qty')

fruits2.plot()

# 3. 선 그래프 옵션 전달
# 1) plot 메서드 내부 옵션 전달 방식(상세 옵션 전달 불가)
# legend의 위치, 글씨 크기 전달 불가

fruits2.plot(xticks,     # x축 눈금
             ylim,       # y축 범위
             fontsize,   # 글자 크기
             rot,        # (x축 이름) 글자 회전 방향
             style,      # 선 스타일
             titie,      # 그래프 이름 
             kind,       # 그래프 종류(기본은 선그래프)
             )


fruits2.plot(xticks = fruits2.index,
             style = '--')

# [ 참고 - 선 스타일 종류 및 전달 ]
# 'r--' : 붉은 대시선
# 'k-'  : 검은색 실선
# 'b.'  : 파란색 점선
# 'ko--' : 검은색 점모양 대시선

s1.plot(style='b.')
s1.plot(color='b', linestyle='--', marker='o')


s1.index = ['월','화','수','목','금','토']
s1.plot()  # x축 이름이 깨짐(한글)

plt.rc('font', family='Malgun Gothic')  # 글씨체 변경

# [ 연습 문제 ]
# cctv.csv를 불러오고 각 년도별 검거율 증가추이를 
# 각 구별로 비교할 수 있도록 plot 도표 그리기
cctv = pd.read_csv('cctv.csv', engine='python')

# 검거율 구하기
cctv['검거율'] = cctv['검거'] / cctv['발생'] * 100

# 교차 테이블 생성
cctv2 = cctv.pivot_table(index='년도', columns='구', values='검거율')

plt.plot(cctv2)
cctv2.plot(title='구별 검거율',
           xticks=cctv2.index,
           rot=30,
           fontsize=8,
           ylim=[0,150],
           style='--')

# 2) 광 범위 옵션 전달 방식 : plt.옵션함수명
# 종류 확인 : dir(plt)

# 2-1) x축, y축 이름
plt.xlabel('발생년도')
plt.ylabel('검거율')

# 2-2) 그래프 제목
plt.title('구별 년도별 검거율 변화')

# 2-3) x축, y축 범위
plt.ylim([0,130])

# 2-4) x축, y축 눈금
plt.xticks(cctv2.index)

# 2-5) legend***
plt.legend(fontsize=6, loc='upper right', title='구 이름')

# 참고 : 선그래프 옵션 확인 방법
plt.plot(s1)   # 선 스타일, 마커 스타일 확인 가능




# 4. barplot 그리기
# - 행 서로 다른 그룹
# - 각 컬럼의 데이터들이 서도 다른 막대로 출력 기본(R에서는 beside=T)
# - 각 컬럼의 데이터들이 하나의 막대로 출력(stacked=True)
# - column 이름이 자동으로 범례 이름으로 전달
# - index 이름이 자동으로 x축 이름으로 전달

fruits2.plot(kind='bar')
plt.xticks(rotation=0)    # x축 눈금 회전

# [ 참고 : plt.rc로 global 옵션 전달 방식 ]
plt.rcParams.keys()       # 각 옵션그룹 별 세부 옵션 정리

plt.rc(group,     # 파라미터 그룹
       **kwargs)  # 상세 옵션

plt.rc('font', family='Malgun Gothic')




# [ 연습 문제 ]
# kimchi_test.csv 파일을 읽고, 
# 각 월별로 김치의 판매량을 비교할 수 있도록 막대그래프로 표현

test3 = pd.read_csv('kimchi_test.csv', engine='python')

test4 = test3.pivot_table(index='판매월', columns='제품',
                          values='수량', aggfunc='sum')

test4.plot(kind='bar',
           ylim=[0,300000],
           rot=0)


plt.legend(title='김치이름', fontsize=7)
plt.ylabel('판매량')
plt.title('월별 김치 판매량 비교')

plt.bar?


# 히스토그램
cctv['CCTV수'].hist(bins=10)                     # 막대의 개수
cctv['CCTV수'].hist(bins=[0,100,200,500,1000])   # 막대의 범위

cctv['CCTV수'].plot(kind='kde')                  # 커널 밀도 함수(누적분포)

