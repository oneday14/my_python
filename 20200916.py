run profile1

# 1. shoppingmall.txt파일을 읽고 쇼핑몰 웹 주소만 출력(총 25개)
# http://.+ 사용 X

# step1) 파일 불러오기
# 1) 하나의 문자열로 만들기
c1 = open('shoppingmall.txt')
test1 = c1.readlines()
c1.close()

# sol1) for문을 사용한 문자열 결합
vstr=''

for i in test1 :
    vstr = vstr + i

# sol2) cat 메서드로 분리된 원소를 하나의 문자열로 결합
vstr2 = Series(test1).str.cat()    

# 2) Series로 만들기
test2 = pd.read_csv('shoppingmall.txt', engine='python', 
                    header=None, sep=';')
test2 = test2.iloc[:,0]
test2.str.findall()

# step2) 패턴 생성
# http://www.wemakeprice.com
# http://ddrago.co.kr.jg 
# http://smartstore.naver.com/wowow312 

import re

p1 = 'http://[a-z0-9./]+'
pat1 = re.compile(p1, flags=re.IGNORECASE)
pat2 = re.compile('http://.+', flags=re.IGNORECASE) # 현재는 가능,
                                                    # 주소 뒤 글자 있을 경우
                                                    # 주소만 추출 어려움

# step3) 패턴 추출
len(pat1.findall(vstr2))
pat2.findall(vstr2)
    
test2.str.findall(pat1).str[0].dropna()
    

# 2. ncs학원검색.txt 파일을 읽고 다음과 같은 데이터 프레임 형식으로 출력
# name        addr        tel         start         end    
# 아이티윌  서울 강남구 02-6255-8001  2018-10-12  2019-03-27
# 아이티윌   ( 서울 강남구 ☎ 02-6255-8002 ) 훈련기관정보보기  훈련기간 : 2018-10-12 ~ 2019-03-27  

# step1) 파일 불러오기
c2 = open('ncs학원검색.txt')
test3 = c2.readlines()
c2.close()

test3 = Series(test3)
test3 = pd.read_csv('ncs학원검색.txt', engine='python', 
                    header=None, sep=';').iloc[:,0]

# step2) 패턴 생성
# 아이티윌   ( 서울 강남구 ☎ 02-6255-8002 ) 훈련기관정보보기  훈련기간 : 2018-10-12 ~ 2019-03-27  

p2 = '(.+) \( (.+) ☎ ([0-9-]+) \) .+ : ([0-9-]+) ~ ([0-9-]+)'
pat3 = re.compile(p2)

# step3) 패턴 추출
c1 = test3.str.findall(pat3).str[0].dropna().str[0].str.strip()
c2 = test3.str.findall(pat3).str[0].dropna().str[1].str.strip()
c3 = test3.str.findall(pat3).str[0].dropna().str[2].str.strip()
c4 = test3.str.findall(pat3).str[0].dropna().str[3].str.strip()
c5 = test3.str.findall(pat3).str[0].dropna().str[4].str.strip()

# step4) DataFrame 생성
DataFrame({'name':c1, 'addr':c2, 'tel':c3, 'start':c4, 'end':c5 })

########## 여기까지는 복습입니다. ##########

# [ 연습 문제 ]
# 교습현황.csv파일을 읽고
# 서울특별시 관악구 남부순환로 1432-7 대우빌딩3층862-5495 (신림동)
# 서울특별시 동작구 성대로29길 79 , 1층 (상도동, 대광빌딩)

df1 = pd.read_csv('교습현황.csv', engine='python', skiprows=1)

# 1) 동별 분기별 총 보습금액 출력
# step1) 동이름 추출
pat5 = re.compile('.+\(([가-힣0-9]+)[),]')
df1['동'] = df1['교습소주소'].str.findall(pat5).str[0].str.replace('[0-9]','')
df1.columns

# step2) multi-index 생성(동, 교습과정)
df1 = df1.set_index(['동', '교습과정'])
df1 = df1.iloc[:,4:]

# step3) 천단위 구분기호 제거 및 숫자 변경
df1 = df1.applymap(lambda x : int(x.replace(',','')))

# step4) 분기 가공
pat6 = re.compile('.+\(([0-9]{1,2})\)')
g1 = df1.columns.str.findall(pat6).str[0].astype(int)
g2 = pd.cut(g1, [1,3,6,9,12],  # [1,3], (3,6], (6,9] ...
            include_lowest=True, 
            labels=['1/4분기','2/4분기','3/4분기','4/4분기'])

# step5) 그룹핑
df1.groupby(level=0).sum().groupby(g2, axis=1).sum()

# 2) 동별 보습액 총 액이 가장 높은 2개의 보습과정을 보습액과 함께 출력
# 1) 동별 교습과정별 총 합
df2 = df1.groupby(level=[0,1]).sum()

# 2) 전체 컬럼 총 합
df2 = df2.sum(1)

# 3) 사용자 정의함수 생성 후 그룹별(동) 전달
f_sort = lambda x : x.sort_values(ascending=False)[:2]
df2.groupby(level=0, group_keys=False).apply(f_sort)


# [ 연습 문제 ]
# oracle_alert_testdb.log 파일을 읽고 다음과 같은 데이터 프레임 생성
# ORA-1109 signalled during: ALTER DATABASE CLOSE NORMAL...
# ORA-00313: open failed for members of log group 1 of thread 1

# code   error
# 00312  online log 1 thread 1

c3 = open('oracle_alert_testdb.log')
test5 = c3.readlines()
c3.close()

pat5 = re.compile('ORA-([0-9:]+) (.+)', flags=re.IGNORECASE)
s1 = Series(test5).str.findall(pat5).str[0].dropna()

t1 = s1.str[0].str.strip().str.replace(':','')
t2 = s1.str[1].str.strip()

df5 = DataFrame({'code':t1, 'error':t2})
df5.groupby('code')['code'].count().sort_values(ascending=False)

df5['error'][df5['code'] == '00600']

