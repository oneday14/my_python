run profile1

# 1. 교습현황.csv 파일을 읽고
test1 = pd.read_csv('교습현황.csv', engine='python', skiprows=1)
test1.columns

# 구 이름 추출
test1['구'] = test1['교습소주소'].map(lambda x : x[6:9])

# 불필요 컬럼 제외
test1 = test1.drop(['교습소주소', '분야구분', '교습계열'], axis=1)

# multi-index 생성(구,교습과정,교습소명)
test1 = test1.set_index(['구','교습과정','교습소명'])

# 년도, 분기, 월 변수 생성
c1 = test1.columns.map(lambda x : x[:4])
c2 = test1.columns.map(lambda x : x[5:].replace(')',''))
c3 = Series(c2).replace(c2[:12],[1,1,1,2,2,2,3,3,3,4,4,4])

# multi-colum 설정(년,분기,월)
test1.columns = [c1,c3,c2]

# 천단위 구분기호 제거 후 숫자 컬럼 변경
test1 = test1.applymap(lambda x : x.replace(',','')).astype('int')

# 1) 교습과정별 분기별 교습 금액의 총 합 출력
test1 = test1 / 1000
test1.sum(axis=0, level=1).sum(axis=1, level=1).stack().reset_index()

# 2) 각 구별, 교습과정별 교습금액의 총 합이 가장 높은 교습소명 출력
test1_1 = test1.sum(axis=1).sum(level=[0,1,2]).unstack().fillna(0)
test1_1.idxmax(1)

# 2. movie_ex1.csv 파일을 읽고(20200730 in R) 
test2 = pd.read_csv('movie_ex1.csv', engine='python')

# 요일 컬럼 생성
d1 = test2['년'].astype('str') + '/' + test2['월'].astype('str') + '/' + test2['일'].astype('str')
test2['요일'] = d1.map(lambda x : 
                       datetime.strptime(x, '%Y/%m/%d').strftime('%A'))

# 1) 연령대별 성별 이용비율의 평균을 구하여라
test2_1 = test2.pivot_table(index='연령대', columns='성별', values='이용_비율(%)',
                            aggfunc='sum')  

test2_1.stack().reset_index()

# 2) 요일별 이용비율의 평균을 구하여라.
test2.pivot_table(index='요일', values='이용_비율(%)')

# 3. delivery.csv 파일을 읽고
test3 = pd.read_csv('delivery.csv', engine='python')

# 1) 일자별 총 통화건수를 구하여라
test3_1 = test3.pivot_table(index='일자', values='통화건수', aggfunc='sum')

# 2) 음식점별 주문수가 많은 시간대를 출력
# 중국음식 12  600
# 보쌈    18   550
#          ...
# ...     24 30
# step1) 교차 테이블 생성
test3_2 = test3.pivot_table(index='시간대', columns='업종', values='통화건수',
                            aggfunc='sum')

# step2) 위 데이터를 조인 가능한 형태(long data)로 변경
test3_2 = test3_2.stack().reset_index()
test3_2 = test3_2.rename({0:'cnt'}, axis=1)

# step3) 업종별 콜수가 많은 시간대 출력
test3_3 = test3_2.idxmax(0).reset_index()
test3_3 = test3_3.rename({0:'시간대'}, axis=1) # 조인 가능한 형태

# step4) 조인
pd.merge(test3_2, test3_3, on=['시간대','업종'])

# 3) 일자별 전일대비 증감률을 구하여라
test3_1

(46081 - 39653) / 39653 * 100          # 16.21

# sol1) index 수정으로 이전 값 가져오기
s1 = list(test3_1.iloc[:-1,0])
s1 = Series(s1, index = test3_1.index[1:])

# 증감률 계산
(test3_1.iloc[:,0] - s1) / s1 * 100

# sol2) 이전 값 가져오는 shift 메서드
s2 = test3_1.iloc[:,0]

s2.shift(periods,    # 몇 번째 이전 값을 가져올지
         freq,       # 날짜 오프셋일 경우 날짜 단위 이동 가능
         axis,       # 기본은 이전 행, 컬럼단위로도 이전값 가져올수있음(axis=1)
         fill_value) # 이전 값이 없을 경우 NA 대신 리턴 값

(s2 - s2.shift(1)) / s2 * 100


# shift 
s2.shift(periods,    # 몇 번째 이전 값을 가져올지
         freq,       # 날짜 오프셋일 경우 날짜 단위 이동 가능
         axis,       # 기본은 이전 행, 컬럼단위로도 이전값 가져올수있음(axis=1)
         fill_value) # 이전 값이 없을 경우 NA 대신 리턴 값

# [ 예제 - card_history.csv 파일일 읽고 shift 사용 ]
card = pd.read_csv('card_history.csv', engine='python')
card = card.set_index('NUM')
card = card.applymap(lambda x : x.replace(',','')).astype('int')

card.shift(1, axis=0)  # 행 단위 이동(이전 값 가져오기)
card.shift(1, axis=1)  # 행 단위 이동

card.shift(-1, axis=0)  # 행 단위 이동(이후 값 가져오기)

########## 여기까지는 복습입니다. ##########

# 문자열 메서드
# - 기본 함수 
# - 문자열 처리와 관련된 함수 표현식
# - 문자열 input, 문자열 output 
# - upper, lower, replace, find, split ....
# - 벡터연산 불가

# 벡터화가 내장된 문자열 메서드 ****
# - pandas 제공
# - 문자열 처리와 관련된 함수 표현식
# - 문자열 input, 문자열 output 
# - upper, lower, replace, find, split ....
# - 벡터연산 가능
# - str 모듈 호출 후 사용
# - Series만 적용 가능, DataFrame 불가

L1 = ['a;b;c', 'A;B;C']
s1 = Series(L1)

L1.split(';')                              # 벡터연산 불가

[i.split(';')[0] for i in L1]              # 리스트 내포 표현식 반복 처리
list(map(lambda x : x.split(';')[0], L1))  # mapping 

s1.split(';')       # Series 객체 적용 불가

# 1. split
s1.str.split(';')                       # Series 객체 적용 불가
s1.str.split(';')[0]                    # split은 벡터연산 가능, 색인은 불가
s1.str.split(';').map(lambda x :x[0])   # 색인 벡터 처리

# 2. replace
s1.replace('a','A')       # 값치환 메서드
s1.str.replace('a','A')   # 벡터화 내장된 문자열 메서드

# =============================================================================
# replace 형태
# 1. 문자열 메서드
# 2. 값치환 메서드
# 3. 벡터화 내장된 문자열 메서드
# =============================================================================

# 3. 대소치환
s1.upper()      # 불가
s1.str.upper()  # 가능
s1.str.lower()  # 가능
s1.str.title()  # 가능

[ i.title() for i in L1 ] 

# =============================================================================
# 참고 : title의 특징
#
# 'abc'.title()       # 'Abc'
# 'abc ncd'.title()   # 'Abc Ncd'
# 'abc;ncd'.title()   # 'Abc;Ncd'
# 
# =============================================================================


# 4. 패턴여부
s1.str.startswith('a')    
s1.str.endswith('a')    

s1.str.startswith('a',1)  # position 전달 의미 X

'a' in 'abc'              # 문자열 포함여부는 in 연산자로 처리
'abc'.contains('a')       # 기본 문자열 메서드로는 불가
s1.str.contains('a')      # 문자열의 포함 여부 전달 가능

# 5. 개수
len('abd')     # 문자열의 크기
len(L1)        # 리스트 원소의 개수(각 원소의 문자열의 크기 X)
s1.str.len()   # 각 원소의 문자열의 크기 리턴

'abcabaa'.count('a')                          # 'a'를 포함하는 횟수
Series(['aa1','abda','a1234']).str.count('a') # 벡터 연산

# 6. 제거함수(strip)
s2 = Series([' abc ', ' abcd', 'abc12 '])
s2.str.strip().str.len()   # 양쪽 공백 제거 확인
s2.str.lstrip().str.len()  # 왼쪽 공백 제거 확인
s2.str.rstrip().str.len()  # 왼쪽 공백 제거 확인

'abd'.lstrip('a')             # 문자 제거
Series('abd').str.lstrip('a') # 벡터화 내장된 메서드 문자 제거 가능

# 7. 위치값 리턴(없으면 -1)
'abdd'.find('d')
Series('abdd').str.find('d')

# 8. 삽입 
a1.pad?     # 문자열 처리 불가

s1.str.pad(width=10,      # 총자리수
           side='both',   # 방향 
           fillchar='-')  # 채울글자

s1.str.pad(10,'both','0')

# 9. 문자열 결합
'a' + 'b' + 'c' 
Series(['a','A']) + Series(['b','B']) + Series(['c','C']) # 벡터연산 가능

s3 = Series(['ab','AB'])
s3.str.cat(sep=';')       # 결합 기호 전달 가능,
                          # Series의 원소를 결합
s3.str.join(sep=';')      # 결합 기호 전달 가능
                          # 원소별 글자 결합

s1.str.split(';').str.cat(sep='')   # 불가
s1.str.split(';').str.join(sep='')  # Series 매 원소마다 결합

# 10. 색인
s1.str.split(';').str[0]            # 벡터화 내장된 색인 처리
s1.str.split(';').str.get(0)        # 벡터화 내장된 메서드 처리 가능


# [ 연습 문제 ]
# professor.csv 파일을 읽고
pro = pd.read_csv('professor.csv', engine='python')

# 1) email-id 출력
vemail = pro.EMAIL.str.split('@').str[0]

# 2) 입사년도 출력
pro.HIREDATE.str[:4]

# 3) ID의 두번째 값이 a인 직원 출력
pro['ID'].str.startswith('a',1)      # 확인 불가(위치값 전달 불가)
pro.loc[pro['ID'].str[1] == 'a', :]

# 4) email_id에 '-' 포함된 직원 출력
pro.loc[vemail.str.contains('-'), :]
pro.loc[vemail.str.find('-') != -1, :]

# 5) 이름을 다음과 같은 형식으로 변경
#    '홍길동' => '홍 길 동'

pro.NAME.str.cat()          # sep='' 가 기본
pro.NAME.str.join()         # sep 옵션 생략 불가
pro.NAME.str.join(sep=' ')  # 각 Series 원소 별 내부 결합 

# 6) PROFNO 컬럼 이름을 PROFID 컬럼으로 변경  
#    (데이터도 4004 => 004004 로 변경)
     
pro = pro.rename({'PROFNO':'PROFID'}, axis=1)
pro.PROFID.astype('str').str.pad(6,'left','0')


card1 = pd.read_csv('card_history.csv', engine='python') 
card = card1.set_index('NUM')

card.str.replace(',','')   # 벡터화 내장된 문자열 메서드는 DataFrame 적용 불가


# 중복값 관련 메서드
# 1) Series 적용
t1 = Series([1,1,2,3,4])

t1.duplicated()       # 내부 정렬 후 순차적으로 이전 값과 같은지 여부 확인
t1[t1.duplicated()]   # 중복값 확인
t1[~t1.duplicated()]  # 중복값 제외

t1.drop_duplicates()

# DataFrame 적용
df1 = DataFrame({'col1':[1,1,2,3,4], 
                 'col2':[1,2,3,4,4],
                 'col3':[2,3,4,4,5]})

df1.drop_duplicates('col1', keep='first')         # 첫 번째 값 남김
df1.drop_duplicates('col1', keep='last')          # 두 번째 값 남김
df1.drop_duplicates('col1', keep=False)           # 중복 값 모두 제거

df1.drop_duplicates(['col1','col2'], keep=False)  # 여러 컬럼 전달 가능

