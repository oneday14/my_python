# -*- coding: utf-8 -*-
run profile1

# SalesJan2009.csv 파일은 온라인 거래 현황에 대한 데이터이다.
sal = pd.read_csv('SalesJan2009.csv', engine = 'python')

# 1.
# Transaction_date은 거래 날짜인데 데이터가 잘못 들어와서
# 올바른 데이터는 1/13/09 5:57 처럼 MM/DD/YY HH24:MI 형식으로 되어 있지만
# 1/02/09 6:17의 날짜가 2001-09-02 6:17 처럼 해석된 데이터도 존재한다.
# 날짜의 포맷을 통일하고 올바르게 파싱되도록 변경한 후,

# step1) 년도, 월, 일, 시간 분리
sal['Month'] = '01'
sal['Year'] = '09'
sal['Day'] = sal['Transaction_date'].str.split('-').str[1].fillna(sal['Transaction_date'].str.split('/').str[1])
sal['Time'] = sal['Transaction_date'].str.split(' ').str[1]

# step2) 년도, 월, 일, 시간 합치기
sal['Transaction_date'] = sal['Month'] + '/' + sal['Day'] + '/' + sal['Year'] + ' ' + sal['Time']

# step3) 날짜 형식으로 변환
sal['Transaction_date'] = sal['Transaction_date'].map(lambda x : datetime.strptime(x, '%m/%d/%y %H:%M'))

# step4) 날짜의 포맷 통일 
sal['Transaction_date'] = sal['Transaction_date'].map(lambda x : datetime.strftime(x, '%m/%d/%y %H:%M'))          ## 결과 나옴


# Transaction_date컬럼 기준 년도별 Payment_Type별 price의 총 합을 교차테이블 형식으로 출력하여라.
# step1) Price컬럼 ',' 제거 및 숫자로 변경 
sal.Price = sal.Price.str.replace(',','').astype('int')

# step2) 교차테이블 생성
sal.pivot_table(index = 'Year',
                columns = 'Payment_Type',
                values = 'Price',
                aggfunc = 'sum')          ## 결과 나옴


# 2.
# student테이블과 exam_01 테이블을 파이썬으로 불러온 후
std = pd.read_csv('student.csv', engine = 'python')
exam = pd.read_csv('exam_01.csv', engine = 'python')

# 1) 각 학생의 정보와 시험성적을 모두 갖는 데이터프레임 생성
df1 = pd.merge(std, exam, on = 'STUDNO')          ## 결과 나옴


# 2) 학년별 성별 시험성적의 평균을 교차테이블 형식으로 출력
# step1) 성별 컬럼 추가
df1['성별'] = Series(np.where(df1.JUMIN.astype('str').str[6] == '1','남', '여'))

# step2) 교차테이블 만들기
df1.pivot_table(index = 'GRADE',
                columns = '성별',
                values = 'TOTAL')          ## 결과 나옴


# 3) 컬럼이름을 모두 소문자로 변경
df1.columns = df1.columns.map(lambda x : x.lower())          ## 결과 나옴


# 4) 학년별 시험성적이 높은순으로 정렬
df1.sort_values(['grade','total'], ascending = [True, False])          ## 결과 나옴


# 5) deptno컬럼 생성, deptno2를 기준으로 하되, 없으면 deptno1 참조
df1['deptno'] = df1.deptno2.fillna(df1.deptno1).astype('int')          ## 결과 나옴


# 6) 학년별 각 학생의 시험성적이 높은 순서대로 순위 출력
s1 = Series([])
for i in range(1,5) :
    s2 = df1.loc[df1.grade == i,'total'].rank(ascending = False)
    s1 = s1.append(s2)
    
df1['순위'] = s1.astype('int')              ## 결과 나옴

