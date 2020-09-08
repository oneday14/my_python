run profile1

# 1. card_history.txt 파일을 읽고
card = pd.read_csv('card_history.txt', sep='\s+')
# sep='\t'  : 탭 분리구분(공백으로 분리된 컬럼을 분리하지 X)         
# sep='\s+' : 한 칸의 공백 이상(탭으로 분리된 컬럼도 분리 가능)

# 인덱스 설정
card = card.set_index('NUM')
   
# 천단위 구분기호 제거 후 숫자 변경
card.replace(',','')     # 값 치환 메서드이므로 ',' 패턴 치환 X
'19,400'.replace(',','') # 치환 가능

card = card.applymap(lambda x : int(x.replace(',','')))

# 1) 각 일별 지출품목의 차지 비율 출력(식료품 : 20%, 의복 : 45%, ....)
card.iloc[0,:] / card.iloc[0,:].sum() * 100
card.iloc[1,:] / card.iloc[1,:].sum() * 100
card.iloc[2,:] / card.iloc[2,:].sum() * 100

f1 = lambda x : x / x.sum() * 100
card.apply(f1, axis=1)

# 2) 각 지출품목별 일의 차지 비율 출력(1일 : 0.7%, 2일 : 1.1%, ....)
card.apply(f1, axis=0)

# 3) 각 일별 지출비용이 가장 높은 품목 출력
# sol1) argmax 사용
card.columns[card.iloc[0,:].argmax()]
card.columns[card.iloc[1,:].argmax()]
card.columns[card.iloc[2,:].argmax()]

f2 = lambda x : card.columns[x.argmax()] 
card.apply(f2, axis=1)

# sol2) idxmax 사용
card.iloc[0,:].idxmax()
card.iloc[1,:].idxmax()
card.iloc[2,:].idxmax()

f3 = lambda x : x.idxmax()

card.apply(f3, axis=1)     # apply를 사용한 각 행별 적용
card.idxmax(axis=1)        # idxmax 자체 행별 적용(axis)

# =============================================================================
# [ 참고 : 최대, 최소를 갖는 index 출력 함수 정리 ] 
#
# s1 = Series([1,3,10,2,5], index=['a','b','c','d','e'])
# 
# # 1. whichmax, whichmin in R
# # 2. argmax, argmin in python(numpy)
# # 3. idxmax, idxmin in python(pandas)
# 
# s1.argmax()  # 위치값 리턴
# s1.idxmax()  # key값 리턴
# 
# =============================================================================

# 4) 각 일별 지출비용이 가장 높은 두 개 품목 출력
s1.sort_values?       # Series 정렬 시 by 옵션 필요 X
card.sort_values?     # DataFrame 정렬 시 by 옵션 필요

card.iloc[0,:].sort_values(ascending=False)[:2].index
card.iloc[1,:].sort_values(ascending=False)[:2].index        # index 리턴
card.iloc[2,:].sort_values(ascending=False)[:2].index.values # 리스트 리턴
Series(card.iloc[2,:].sort_values(ascending=False)[:2].index) # 시리즈 리턴

f4 = lambda x : x.sort_values(ascending=False)[:2].index
f4 = lambda x : x.sort_values(ascending=False)[:2].index.values
f4 = lambda x : Series(x.sort_values(ascending=False)[:2].index)

card.apply(f4, axis=1)

# 2. 'disease.txt' 파일을 읽고 
df1 = pd.read_csv('disease.txt',sep='\s+', engine='python')
df1.index
# 1) 월별 컬럼 인덱스 설정
# 2) index와 column 이름을 각각 월, 질병으로 저장
# 3) NA를 0으로 수정
# 4) 대장균이 가장 많이 발병한 달을 출력
# 5) 각 질병 별 발병횟수의 총 합을 출력

# 3. employment.csv 파일을 읽고
emp2 = pd.read_csv('employment.csv', engine='python')

# 1) 년도와 각 항목(총근로일수, 총근로시간...)을 멀티 컬럼으로 설정
# index 설정
emp2 = emp2.set_index('고용형태')

# 멀티 컬럼 설정
# step1) 현재 컬럼 이름 가공(2007.1 => 2007)
c1 = emp2.columns.map(lambda x : x[:4])    # 1 level value

# step2) 현재 첫번째 행 단위 제거(월급여액 (천원) => 월급여액)
c2 = emp2.iloc[0,:].map(lambda x : x.split(' ')[0])

# step3) c1, c2 멀티 컬럼 전달
emp2.columns = [c1,c2]

# step4)  첫번째 행 제거
emp2 = emp2.drop('고용형태', axis=0)

# 2) 모두 숫자 컬럼으로 저장('-'는 0으로 치환)
# '-'를 '0'으로 치환
emp2.applymap(lambda x : x.replace('-','0'))
emp2 = emp2.replace('-',0)

# ',' 제거
emp2 = emp2.applymap(lambda x : str(x).replace(',',''))

# 숫자 변경
emp2 = emp2.astype('float')

########## 여기까지는 복습입니다. ##########

# 멀티 인덱스
# 1. 생성
df22 = DataFrame(np.arange(1,17).reshape(4,4),
                 index = [['A','A','B','B'], ['a','b','a','b']],
                 columns = [['col_a','col_a','col_b','col_b'],
                            ['col1','col2','col1','col2']])

# 2. 색인
df22['col_a']   # 상위 컬럼의 key 색인 가능
df22['col1']    # 하위 컬럼 에러발생
df22.iloc[:,0]  # 멀티 인덱스의 위치값 색인 가능

df22.loc[:,'col_a'] # 상위 컬럼의 이름 색인 가능
df22.loc[:,'col1']  # 하위 컬럼의 이름 색인 불가
df22.loc['A',:]     # 상위 인덱스의 이름 색인 가능
df22.loc['a',:]     # 하위 인덱스의 이름 색인 불가

# 멀티 인덱스 색인 메서드 : xs(함수)
df22.iloc[]
df22.loc[]
df22.xs('col2', axis=1, level=1).xs('b', axis=0, level=1)

df22.loc[:, ('col_a','col1')]          # 상위부터 하위까지의 순차적 색인은 iloc 가능
df22.loc[('A','a'), ('col_a','col1')]  # 상위부터 하위까지의 순차적 색인은 iloc 가능


# [ 연습 문제 ]
# multi_index.csv 파일을 읽고 멀티 인덱스를 갖는 데이터프레임 변경
df33 = pd.read_csv('multi_index.csv', engine='python')
df33.iloc[:,0] = df33.iloc[:,0].fillna(method='ffill')
df33 = df33.set_index(['Unnamed: 0','Unnamed: 1'])
df33.index.names = ['지역','지점']
df33.columns =['냉장고','냉장고','TV','TV']
df33.columns = [df33.columns, df33.iloc[0,:]]
df33 = df33.iloc[1:,]
df33.columns.names = ['품목','구분']

# 1) 모든 품목의 price 선택
df33.iloc[:,[0,2]]
df33.xs('price', axis=1, level=1)

# 2) A 지점의 price 선택
df33.iloc[[0,2],0]
df33.xs('A', axis=0, level=1).xs('price', axis=1, level=1)

# 3) seoul의 'B' 지점 선택
df33.iloc[1,:]
df33.loc[('seoul','B'), :]

# 4) 냉장고의 price 선택
df33['냉장고']['price']
df33.iloc[:,0]
df33.loc[:,('냉장고','price')]

# 5) 냉장고의 price, TV의 qty 선택
df33.iloc[:,[0,3]]
df33.loc[:,('냉장고','price')]
df33.loc[:,('TV','qty')]
df33.loc[:,[('냉장고','price'),('TV','qty')]] # ****

# [ 연습 문제 ]
# 다음의 데이터 프레임을 멀티 인덱스 설정 후
df1 = pd.read_csv('multi_index_ex1.csv',encoding='cp949')

# 인덱스 설정
df1 = df1.set_index(['지역','지역.1'])
df1.index.names = ['구분','상세']

# 컬럼 설정
c1 = df1.columns.map(lambda x : x[:2])
df1.columns = [c1, df1.iloc[0,:]]
df1 = df1.iloc[1:, :]              # 첫번째 행 제거
df1.columns.names = ['지역','지점']

# 1) 컴퓨터의 서울지역 판매량 출력
df1.loc['컴퓨터','서울']

# 2) 서울지역의 컴퓨터의 각 세부항목별 판매량의 합계 출력
df1.dtypes
df1 = df1.astype('int')
df1.loc['컴퓨터','서울'].sum(1)

# 3) 각 지역의 A지점의 TV 판매량 출력
df1.loc[('가전','TV'),[('서울','A'),('경기','A'),('강원','A')]].sum()
df1.xs('A', axis=1, level=1).xs('TV', axis=0, level=1).iloc[0,:].sum()

# 4) 각 지역의 C지점의 모바일의 각 세부항목별 판매량 평균 출력
df1.xs('C', axis=1, level=1).loc['모바일',:].mean(1)

# 5) 서울지역의 A지점의 노트북 판매량 출력
df1.loc[:,('서울','A')].xs('노트북', level=1)
df1.xs('A', axis=1, level='지점')

# 3. 산술 연산
# 1) multi-index의 axis만 전달 시 : multi-index 여부와 상관없이 
#    axis=0 : 행별(세로방향)
#    axis=1 : 컬럼별(가로방향)
   
df1.sum(axis=0)
df1.sum(axis=1)

# 2) multi-index의 axis, level 동시 전달 시
#    multi-index의 각 레벨이 같은 값끼리 묶여 그룹 연산 

# 예) 지역별 판매량 총 합
df1.sum(axis=1, level=0)

# 예) 구분별(컴퓨터,가전,모바일) 판매량 총 합
df1.sum(axis=0, level=0)

# [ 연습 문제 ]
# employment.csv 파일을 읽고 멀티인덱스 설정 후(emp2)
emp2

# 1) 각 년도별 정규근로자와 비정규근로자의 월급여액의 차이 계산
s1 = emp2.xs('월급여액', axis=1, level=1).loc['정규근로자',:]
s2 = emp2.xs('월급여액', axis=1, level=1).loc['비정규근로자',:]

s1 - s2

# 2) 각 세부항목의 평균(총근로일수, 총근로시간)
emp2.mean(axis=1, level=1)



# 4. 정렬
# 1) index 순 정렬

df1.sort_index(axis=0, level=0)  # 구분 순서
df1.sort_index(axis=1, level=0)  # 지역 순서
df1.sort_index(axis=1, level=1, ascending=False)  # 지역 순서 역순
df1.sort_index(axis=1, level=[0,1], ascending=[True, False])  # 지역 순서 역순

# 2) 특정 컬럼 값 순 정렬 : 컬럼의 이름을 튜플로 전달
df1.sort_values(by=('서울','A'), ascending=False)
df1.sort_values(by=[('서울','A'),('경기','B')], ascending=False)

# 5. level 치환
card.T
card.swapaxes(1,0)

df1.swaplevel(1,0, axis=1)

# [ 연습 문제 ]
# df1에서 컬럼의 두 레벨을 치환하여 지점이 상위 레벨로 가도록 전달
# A             B            C
# 서울 경기 강원 서울 경기 강원 서울 경기 강원
df1_1 = df1.sort_index(axis=1, level=[1,0], ascending=[False, True])
df1_1.swaplevel(0,1,axis=1)

# [ 연습 문제 ]
# 다음의 데이터프레임을 읽고 날짜, 지점, 품목의 3 level index 설정 후
sales = pd.read_csv('sales2.csv', engine='python')

# 인덱스 설정
sales = sales.set_index(['날짜','지점','품목'])

# 1) 출고 컬럼이 높은 순서대로 정렬
sales.sort_values(by='출고', ascending=False)

# 2) 품목 인덱스를 가장 상위 인덱스로 배치
sales.swaplevel(0,2,axis=0)  # 정렬 필요
sales.sort_index(axis=0, level=[2,1]).swaplevel(0,2,axis=0)  # 정렬 필요
