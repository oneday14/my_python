# group by 기능
# 1. index값이 같은 경우 그룹 연산
# 2. wide 형태(행별, 열별 그룹 연산 가능)
# 3. long 형태(pivot_table)
# 4. group_by

# groupby 메서드
# - 분리-적용-결합
# - 특정 컬럼의 값이 같은 경우 grouping(기본 방향)
# - 행별, 컬럼별 grouping 가능
# - tidy 형식(long data)의 데이터에 적용 가능
# - groupby 함수 안에 그룹함수를 전달하는 방식이 X

emp = pd.read_csv('emp.csv')

# 1) pivot_table
emp.pivot_table(index='DEPTNO', values='SAL', aggfunc='sum')

# 2) index 생성 후
emp.set_index('DEPTNO')['SAL'].sum(axis=0, level=0)

# 3) groupby
emp.groupby('DEPTNO')              # 분리만 수행
emp.groupby('DEPTNO').sum()        # 연산 가능한 모든 컬럼에 대해 그룹연산
emp.groupby('DEPTNO')['SAL'].sum() # 선택된 컬럼에 대해서만 그룹연산

# 2) 여러 연산 컬럼 전달
emp.groupby('DEPTNO')['SAL','COMM'].mean()

# 3) 여러 groupby 컬럼 전달
emp.groupby(['DEPTNO','JOB'])['SAL'].mean()

# 4) 여러 함수 전달 : agg(결합 함수)
emp.groupby(['DEPTNO','JOB'])['SAL'].agg(['mean','sum'])
emp.groupby(['DEPTNO','JOB'])[['SAL','COMM']].agg({'SAL':'mean',
                                                   'COMM':'sum'})
                                                  
# 예제) emp 데이터에서 deptno별 sal의 평균
emp.groupby('DEPTNO')['SAL'].sum()   # 연산 컬럼 미리 호출 방식
emp['SAL'].groupby(emp['DEPTNO']).sum()   # 연산 컬럼 미리 호출 방식


# 여러 가지 groupby의 옵션
# 1) as_index : groupby 컬럼의 index 전달 여부(기본 : True)
emp.groupby('DEPTNO')['SAL'].sum().reset_index()
emp.groupby('DEPTNO', as_index=False)['SAL'].sum()

# 2) axis(방향 선택), level(multi-index의 depth)
emp2 = emp.sort_values(by=['DEPTNO','EMPNO']).set_index(['DEPTNO','EMPNO'])

emp2['SAL'].sum(axis=0, level=0)
emp2.groupby(axis=0, level=0)['SAL'].sum()


# 3) 객체를 groupby 컬럼으로 전달
df1 = DataFrame(np.arange(1,17).reshape(4,4),
                index=['a','b','c','d'],
                columns=['A','B','C','D'])

df1.groupby(['g1','g2','g1','g2'], axis=0).sum()
df1.groupby(['g1','g2','g1','g2'], axis=1).sum()

# 4) group_keys : groupby 컬럼의 재출력 방지
#   (groupby 연산 후 연산결과에 groupby 컬럼을 포함하는 경우 생략 가능)

emp.groupby('DEPTNO', group_keys=False)['SAL'].sum()

# [ 연습 문제 ]
std = pd.read_csv('student.csv', engine='python')
exam = pd.read_csv('exam_01.csv', engine='python')

std2 = pd.merge(std, exam).loc[:,['NAME','GRADE','TOTAL']]

# 1. 각 학년별 평균 시험성적
std2.pivot_table(index='GRADE', values='TOTAL')
std2.groupby('GRADE')['TOTAL'].mean()            # Series 출력
std2.groupby('GRADE')[['TOTAL']].mean()          # DataFrame 출력

# 2. 각 학년별, 성별 시험성적의 최대, 최소값
std.JUMIN.astype('str').str[6].replace(['1','2'], ['남자','여자'])
std2['G1'] = std.JUMIN.astype('str').str[6].map({'1':'남자', '2':'여자'})

std2.groupby(['GRADE','G1'])[['TOTAL']].max()              # DataFrame 출력
std2.groupby(['GRADE','G1'])[['TOTAL']].min()              # DataFrame 출력

std2.groupby(['GRADE','G1'])[['TOTAL']].agg(['min','max']) # DataFrame 출력



# [ 참고 - 특정 컬럼 하나 선택 시 차원 축소 방지 ]
std.iloc[:,0]                # Series 리턴
std.iloc[:,0:1]              # DataFrame 리턴(숫자 슬라이스)
std.loc[:,'STUDNO']          # Series 리턴
std.loc[:,'STUDNO':'STUDNO'] # DataFrame 리턴(문자 슬라이스)

std['STUDNO']                # 하나 key 색인 시 Series
std['STUDNO':'STUDNO']       # key indexing에서 slice 색인 불가
std[['STUDNO']]              # key indexing을 사용한 차원 축소 방지***



# [ 연습 문제 ]
# 1. sales3.csv 데이터를 불러와서 
sales3 = pd.read_csv('sales3.csv', engine='python')

# 1) 각 날짜별 판매량의 합계를 구하여라.
sales3.groupby('date')['qty'].sum()

# 2) 각 code별 판매량의 합계를 구하여라.
sales3.groupby('code')['qty'].sum()

# 3) product 데이터를 이용하여 각 날짜별, 상품별 매출의 합계를 구하여라
product = pd.read_csv('product.csv')

sales3_1 = pd.merge(sales3, product)
sales3_1['total'] = sales3_1['qty'] * sales3_1['price']
sales3_1.groupby(['date','product'])['total'].sum()

# 2. emp 데이터에서 각 연봉의 등급별 연봉의 평균을 출력
# 단, 연봉의 등급은 3000이상 A, 1500 이상 3000미만 B, 1500미만 C
# [0,1500) , [1500,3000), [3000, 10000)

g1 = np.where(emp.SAL >= 3000, 'A',
                               np.where(emp.SAL >= 1500, 'B', 'C'))

pd.cut(emp['SAL'], 
       bins=[0, 1500, 3000, 10000], 
       right=False,
       labels=['C','B','A'])

emp['SAL'].groupby(g1).mean()


# cut
# - binding 작업

pd.cut(x,                     # 실제 대상(1차원)
       bins,                  # cutting 구간 나열
       right=True,            # 오른쪽 닫힘 여부 (1,2], (2,3], (3,4]
       labels,                # cutting 객체에 이름 부여
       include_lowest=False)  # 최소값 포함 여부

s1 = Series([1,2,3,4,5,6,7,8,9,10])

pd.cut(s1, bins=[1,5,10])     # (1,5], (5,10] 
pd.cut(s1, bins=[1,5,10], labels=('g1','g2'))     
pd.cut(s1, bins=[1,5,10], labels=('g1','g2'), include_lowest=True)




# =============================================================================
# 변수의 변경(binding)
# 1. 연속형 변수를 factor형으로 변경***
# 2. 여러 변수와의 상호작용 고려***
# 
# 학습의 효과
# 시험성적 ~ 학습량X집중력 (interaction)***
# 
# 
# 시험성적 ~ 학습량형태(binding)***
# 
# 학습량(0~10)
# 학습량(11~20)
# 학습량(21~30)
# =============================================================================


# [ 연습 문제 ] 
# subway2.csv 파일을 읽고 각 역별 승하차의 오전/오후별 인원수를 출력
pd.read_csv('subway2.csv', engine='python')

# 데이터의 결합
# 1. append(행 결합)
# 2. merge(컬럼 결합)
# 3. concat(행, 컬럼 결합)
# - 분리되어진 데이터의 union, join 처리
# - 상하결합(axis=0, 기본), 좌우결합(axis=1) 가능
# - join 처리 시 outer join이 기본

df1 = DataFrame({'col1':[1,2,3,4], 'col2':[10,20,30,40]})
df2 = DataFrame({'col1':[1,2,3,4], 'col3':['a','b','c','d']})
df3 = DataFrame({'col1':[5,6], 'col2':[50,60], 'col3':['e','f']})
df4 = DataFrame({'col1':[1,2,3,4], 'col3':['a','b','c','d']},
                index = [0,1,2,4])

df12 = pd.merge(df1, df2, on='col1')
df12.append(df3, ignore_index=True)

pd.concat([df1, df2])          # 세로 방향으로 결합(append 처리, 같은 컬럼끼리)
pd.concat([df1, df2], axis=1)  # 가로 방향으로 결합(index로 join 처리)
                               # merge와는 다르게 중복된 컬럼 생략 X

df12 = pd.concat([df1, df2], axis=1).iloc[:,[0,1,3]]
pd.concat([df12, df3], ignore_index=True)


# df1과 df4를 join
pd.merge(df1, df4, on='col1')
pd.merge(df1, df4, left_index=True, right_index=True) # inner join
pd.concat([df1, df4], axis=1)                         # outer join


# [ 연습 문제 ]
# 다음의 데이터를 결합하세요 (emp_1과 emp_2, emp_3)
emp_1 = pd.read_csv('emp_1.csv')
emp_2 = pd.read_csv('emp_2.csv')
emp_3 = pd.read_csv('emp_3.csv')

emp_12 = pd.merge(emp_1, emp_2, on='EMPNO')

emp_12.append(emp_3, ignore_index=True)
pd.concat([emp_12, emp_3], axis=0, ignore_index=True)



# 데이터 입출력
# 1. read_csv
pd.read_csv(file,        # 파일명
            sep=',',     # 분리구분기호
            header=True, # 첫번째 행 컬럼화 여부, None 설정 시 value로 전달
            names,       # 컬럼이름 변경
            index_col,   # index로 설정할 컬럼이름 전달(multi 가능)***
            usecols,     # 불러올 컬럼 리스트
            dtype,       # 불러올 컬럼의 데이터 타입 지정(딕셔너리 형태)
            engine,   
            skiprows,    # 제외할 행 전달
            nrows,       # 불러올 행 개수 전달
            na_values,   # NA 처리 문자열 전달
            parse_dates, # 날짜 파싱처리할 컬럼 전달***
            chunksize,   # 파일을 행 단위로 분리해서 불러올 경우 사용
            encoding)    # 인코딩 옵션

pd.read_csv('read_test.csv', header=None)
pd.read_csv('read_test.csv').dtypes                # date 컬럼이 숫자 형식

pd.read_csv('read_test.csv', parse_dates=['date']) # date 컬럼이 날짜 형식

pd.read_csv('read_test.csv', usecols=['date','a']) # 컬럼 선택 가능

pd.read_csv('read_test.csv', dtype='str')          # 전체 컬럼 데이터 타입 변경
pd.read_csv('read_test.csv', dtype={'c':'str'})    # 특정 컬럼 데이터 타입 변경

pd.read_csv('read_test.csv', index_col = 'date')   # 인덱스 컬럼 지정

pd.read_csv('read_test.csv', na_values=[',','.','!','?','-']) 
pd.read_csv('read_test.csv', na_values={'a' : ['.','-'],
                                        'b' : ['?','!']}) 


pd.read_csv('read_test.csv', names=['date','A','B','C','D']) # header 사라짐

pd.read_csv('read_test.csv', nrows=5)      # 컬럼 제외, 불러올 행의 수
pd.read_csv('read_test.csv', skiprows=5)   # 컬럼 포함, 제외할 행의 수
pd.read_csv('read_test.csv', skiprows=[5]) # 제외할 행 번호 

df_test = pd.read_csv('read_test.csv', chunksize=30) 
df_test            # fetch X

# fetch 방법
# 1) 불러올 행의 수 지정, 차례대로 fetch
df_test = pd.read_csv('read_test.csv', chunksize=30) 

df_test1 = df_test.get_chunk(10)
df_test2 = df_test.get_chunk(10)
df_test3 = df_test.get_chunk(10)

# 2) for문을 사용한 print
df_test = pd.read_csv('read_test.csv', chunksize=30) 

for i in df_test :
    print(i)


# 3) for문을 사용하여 하나의 데이터프레임으로 결합***
df_test = pd.read_csv('read_test.csv', chunksize=30)  

df_new = DataFrame()
    
for i in df_test :
    df_new = pd.concat([df_new, i], axis=0)



# 2. read_excel
pd.read_excel('emp_1.xlsx', 'Data')

# 3. read_clipboard
pd.read_clipboard()

    
# [ 연습 문제 - 다음의 사용자 정의 함수 생성 ]
# 모듈에서 함수 찾기    
# find_func('pd', 'excel')
read_excel

dir(pd)

import pandas

s1 = Series(dir(pandas))
s1[s1.str.contains('read')]

def find_func(module, function) :
    s1 = Series(dir(module))
    return list(s1[s1.str.contains(function)])
    
find_func(pandas, 'excel')




    
    
    





