run profile1

# 1. subway2.csv 파일을 읽고 
sub = pd.read_csv('subway2.csv', engine='python', skiprows=1)

# 역 이름 채우기
sub['전체'] = sub['전체'].fillna(method='ffill')

# 전체, 구분 index 생성
sub2 = sub.set_index(['전체','구분'])

# 컬럼 이름 변경
c1 = sub2.columns.str[:2].astype('int')
sub2.columns = c1


# 1) 각 역별 승하차의 오전/오후별 인원수를 출력
g1 = np.where(c1 < 12, '오전', '오후')              # 24시 오후에 포함
g2 = pd.cut(c1, bins=[0,12,24],                    # (0,12], (12,24]
            right=False,                           # [0,12), [12,24)
            labels=['오전', '오후']).fillna('오전')  # 24시 오전에 포함

sub2.groupby(g1,axis=1).sum()
sub2.groupby(g2,axis=1).sum()

# 2) 각 시간대별 승차인원이 가장 큰 5개의 역이름과 승차인원을 함께 출력
sub3 = sub2.xs('승차', level=1)

# 1) 교차테이블로 부터 상위 5개 역이름 추출
sub3.iloc[:,1].sort_values(ascending=False)[:5]  # 5시 시간대 확인

f2 = lambda x : x.sort_values(ascending=False)[:5]
sub4 = sub3.apply(f2, axis=0).stack().sort_index(level=[1,0]).swaplevel(0,1)

f3 = lambda x : x.sort_values(ascending=False)
sub4.groupby(level=0).apply(f3)

# 2) groupby 결과(stack처리된)로 부터 상위 5개 역이름 추출
sub4 = sub3.stack().sort_index(level=[1,0]).swaplevel(0,1)
sub4.groupby(level=0, group_keys=False).apply(f2)

# 2. kimchi_test.csv 파일을 읽고
kimchi = pd.read_csv('kimchi_test.csv', engine='python')

# 1) 각 년도별 제품별 판매량과 판매금액의 평균
kimchi.groupby(['판매년도', '제품'])[['수량','판매금액']].mean()

# 2) 각 년도별 제품별 판매처별 판매량과 판매금액 평균
kimchi.groupby(['판매년도', '제품', '판매처'])[['수량','판매금액']].mean()

# 3) 각 김치별로 가장 많이 팔리는 월과 해당 월의 판매량을 김치이름과 함께 출력
kimchi2 = kimchi.groupby(['제품','판매월'])['수량'].sum()   # Series 리턴
kimchi3 = kimchi.groupby(['제품','판매월'])[['수량']].sum() # DataFrame 리턴

# sol1) multi-index의 idxmax와 색인을 통한 값 추출
kimchi2.groupby(level=0).idxmax()    # Series 리턴
kimchi3.groupby(level=0).idxmax()    # DataFrame 리턴

kimchi2.loc[('무김치', 3)]            # multi-index의 색인

kimchi2.loc[kimchi2.groupby(level=0).idxmax()]  # Series의 값을 색인에 전달
kimchi3.loc[kimchi3.groupby(level=0).idxmax()]  # 에러발생, 수량 키 해석 불가
kimchi3.loc[kimchi3.groupby(level=0).idxmax()['수량']]  # 키 제거 후 가능

kimchi2.loc[[('총각김치',1), ('총각김치',2)]]

# sol2) 정렬 후 판매량이 가장 많은 한 행 출력
kimchi2.xs('총각김치',level=0).sort_values(ascending=False)[:1]

f1 = lambda x : x.sort_values(ascending=False)[:1]

kimchi2.groupby(level=0).apply(f1)                   # groupby 컬럼 중복
kimchi2.groupby(level=0, group_keys=False).apply(f1) # groupby 컬럼 중복 X

# sol3) 교차표의 idxmax와 조인
kimchi4 = kimchi.pivot_table(index='판매월', columns='제품', 
                             values='수량', aggfunc='sum')

kimchi5 = kimchi4.idxmax(axis=0).reset_index().rename({0:'월'}, axis=1)
kimchi2_1 = kimchi2.reset_index()
pd.merge(kimchi2_1, kimchi5, left_on=['제품','판매월'],
                             right_on=['제품','월'])

# 3. delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine='python',
                   parse_dates=['일자'])

# 1) 요일별로 각 업종별 통화건수 총 합 확인
# 요일 출력
deli['요일'] = deli['일자'].map(lambda x : x.strftime('%A'))

deli.groupby(['요일','업종'])['통화건수'].sum()

# 2) 평일과 주말(금,토,일) 각 그룹별 시군구별 통화건수 총 합 출력
d1 = {'Monday':'평일', 'Tuesday':'평일', 'Wednesday':'평일',
      'Thursday':'평일', 'Friday':'주말','Saturday':'주말','Sunday':'주말'}
d2 = {['Monday','Tuesday','Wednesday','Thursday'] :'평일', 
      ['Friday','Saturday','Sunday']:'주말'} # 여러 key를 갖는 딕셔너리 생성 불가

g3 = deli['요일'].map(d1)
deli.groupby([g3, '시군구'])['통화건수'].sum()

########## 여기까지는 복습입니다. ##########

# [ 연습 문제 ]
# taxi_call.csv 데이터를 사용하여
taxi = pd.read_csv('taxi_call.csv', engine='python')

# 1) 구별 택시콜이 가장 많은 시간대와 콜 수 함께 출력
taxi2 = taxi.groupby(['발신지_시군구', '시간대'])['통화건수'].sum()
g1 = taxi2.groupby(level=0).idxmax()

taxi2.loc[g1]

# 2) 다음의 시간대별 통화건수의 총 합 출력 
#    20 ~ 03시(야간), 03 ~ 08시(심야), 08 ~ 15시(오전), 15 ~ 20(오후)
b1 = [20,3,8,15,20]
b2 = [0,3,8,15,20,24]

pd.cut(taxi['시간대'], bins=b1)  # 한 방향 증가, 감소값만 bins로 전달 가능
pd.cut(taxi['시간대'], bins=b2)  # 한 방향 증가, 감소값만 bins로 전달 가능

c1 = pd.cut(taxi['시간대'], bins=b2, 
            include_lowest=True,   # (-0.001, 3.0], (3,8]
            labels=['야간1','심야','오전','오후','야간2'])

c1.replace('야간1','야간')                   # 한개값 치환 가능
c1.replace(['야간1','야간2'],'야간')          # old value만 리스트 전달 X 
c1 = c1.replace(['야간1','야간2'],['야간','야간']) # new value도 리스트 전달 

taxi['통화건수'].groupby(c1).sum()

# 3) 구별 택시콜이 가장 많은 읍면동 상위 3개와 콜수 함께 출력
taxi3 = taxi.groupby(['발신지_시군구','발신지_읍면동'])['통화건수'].sum()

f_sort = lambda x : x.sort_values(ascending=False)[:3]
taxi3.groupby(level=0, group_keys=False).apply(f_sort)


# 정규식 패턴
# ^ : 시작
# $ : 끝 
# . : 하나의 문자
# [] : 여러개 문자 조합 ex) '[0-9]'
# \ : 특수기호 일반기호 화
# () : group 형성 기호


# replace 메서드와 정규식 표현식의 사용
s1 = Series(['ab12','abd!*0','abc'])
s2 = Series([1,2,3,4,5])
s3 = Series(['abcd','bcdf','Abc'])

# 1. 문자열 메서드 : 사용 불가
'ab12'.replace('[0-9]','')
'ab[0-9]12'.replace('[0-9]','')

# 2. 값 치환 메서드 : 전달 가능
s2.replace('[3-5]','')               # 전달 불가
s2.replace('[3-5]','', regex=True)   # 전달 불가

s3.replace('^a', value='***') # 전달 불가
s3.replace(to_replace=r'^a...', value='***', regex=True) # 전달 가능

s2.replace([3,4,5],['a','b','c'])  # 리스트 대 리스트 매핑 치환 가능

# 3. 벡터화 내장된(str.replace) 메서드 : 사용 가능
s1.str.replace('[0-9]','')  # 전달 가능
s1.str.replace


# 정규식 표현식을 사용한 함수

# 1. findall
# - 정규식 표현식에 매칭되는 값을 추출
# - 벡터 연산 불가
# - str.findall로 벡터 연산 처리 가능(Series에 적용 가능)
# - 정규식 표현식은 re.complie로 미리 complie 가능(메모리 절감)


vemail = '''IJ U 12 abc@naver.com 1234 ! abc a123@hanmail.net JHHF 
         ! aa12@daum.net jgjg 3333 ***'''

vemail2 = ['IJ U 12 abc@naver.com 1234 !', 
          'abc a123@hanmail.net JHHF', 
          '! aa12@daum.net.kr jgjg 3333 ***']

#  정규식 표현식의 compile
import re            # 정규식 표현식 호출 모듈
re.compile(pattern,  # parsing할 정규식 표현식
           flags)    # 기타 전달(대소 구분 같은거 전달)

r1 = re.compile('[a-z0-9]+@[a-z]+.[a-z]{1,3}',  flags = re.IGNORECASE) 

# findall로 패턴에 매칭되는 문자열 추출
r1.findall(vemail)             # 문자열의 정규식 표현식 추출 가능
r1.findall(vemail2)            # 벡터 연산 불가
Series(vemail).str.findall(r1) # Series에서 str.findall로 벡터 연산 가능


# [ 연습 문제 ]
# 다음의 문자열에서 문자+숫자 형식으로 된 단어만 추
str1 = '''adljf abd+123 fieij 1111 abc111 Ac+0192 jknkj
          lkjl asdf+0394 jjj'''
         
r2 = re.compile('[a-z]+\+[0-9]+',  flags = re.IGNORECASE) 
r2.findall(str1)          
          
# 정규식 표현을 사용한 그룹핑(findall로 각 그룹 추출)  
r3 = re.compile('([a-z0-9]+)@[a-z]+.[a-z]{1,3}',  flags = re.IGNORECASE)         
r4 = re.compile('([a-z0-9]+)@([a-z]+).([a-z]{1,3})',  flags = re.IGNORECASE)         

r1.findall(vemail)
r3.findall(vemail)

t1 = Series(r4.findall(vemail)).str[0]
t2 = Series(r4.findall(vemail)).str[1]
t3 = Series(r4.findall(vemail)).str[2]

DataFrame({'t1':t1,'t2':t2,'t3':t3})


# [ 참고 ]
vstr2='abc@naver.com'

vemail
r5 = re.compile('.+@.+',  flags = re.IGNORECASE) 
r5.findall(vstr2)    # 공백을 포함하지 않는 문자열에서는 이메일 주소 추출 잘됌
r5.findall(vemail)   # 공백을 포함하는 경우는 .의 전달은 X

