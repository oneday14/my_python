run profile1

# 1. 아래와 같은 데이터 프레임 생성 후(세 개의 컬럼을 갖는)
# name price qty
# apple 2000 5
# mango 1500 4
# banana 500 10
# cherry 400 NA
d1 = {'name':['apple','mango','banana','cherry'],
      'price':[2000,1500,500,400],
      'qty':[5,4,10,NA]}

df1 = DataFrame(d1)

# =============================================================================
# python의 색인 방식
#
# 1. 기본 색인 R과 유사 : [row index, column index]
# 2. numpy의 리스트 색인 시 : [[row index], [column index]] 불가
#       => [np.ix_([row index], [column index])]
# 3. pandas 색인 : [row index, column index] 불가
#       => df1.iloc[row index, column index] # 위치기반 색인
#          df1.loc[row index, column index]  # 이름기반 색인(조건 가능)
# =============================================================================
         
# 1) mango의 price와  qty 선택
df1.iloc[1:2,[1,2]]  # 차원축소 방지 => DataFrame으로 출력
df1.loc[df1.name == 'mango',['price','qty']]

# 2) mango와 cherry 의 price 선택(위치색인 불가)
df1.iloc[[1,3], 1:2]
df1.loc[(df1['name'] == 'mango') | (df1['name'] == 'cherry'), 'price']


# =============================================================================
# pandas에서의 in 연산자 : isin**
# 
# df1['name'][0] in ['mango','cherry']  # 스칼라에 대해 in 연산자 처리 가능
# df1['name'] in ['mango','cherry']     # 시리즈에 대해 in 연산자 벡터연산 불가
# 
# df1['name'].map(lambda x : x in ['mango','cherry']) # in연산자 mapping 처리
# df1['name'].isin(['mango','cherry'])                # isin 메서드 처리
# 
# =============================================================================

# 3) 전체 과일의 price만 선택
df1.iloc[:,1]        # Serie 출력
df1.iloc[:,1:1]      # data 출력 X(1에서 0까지 출력)
df1.iloc[:,1:2]      # DataFrame 출력(1에서 1까지 출력)

df1.loc[:,'price']   # Series 출력
df1['price']         # Series 출력

df1.loc[:,'price':'qty']    # DataFrame 출력, qty 까지 출력
df1.loc[:,'price':'price']  # DataFrame 출력, price만 출력

# =============================================================================
# [ 참고 - 슬라이스 색인의 형태]
# 
# n:m => n에서 (m-1) 까지 추출(마지막 범위 미포함)
# name1:name2 => name1에서 name2까지 추출(마지막 범위 포함)
# 
# =============================================================================

# 4) qty의 평균
df1.qty.mean()
np.mean(df1.qty)

# 5) price가 1000 이상인 과일 이름 출력
df1.loc[df1.price >= 1000, 'name']        # Series 출력
df1.loc[df1.price >= 1000, 'name':'name'] # DataFrame 출력

# 6) cherry, banana, mango, apple 순 출력
# 1) 위치값 기반
df1.iloc[[3,2,1,0]]

# 2) 이름 기반 reindexing
df1.index = df1.name
df1 = df1.iloc[:,1:]  # name 제외

df1.loc[['cherry', 'banana', 'mango', 'apple'],:]
DataFrame(df1, index=['cherry', 'banana', 'mango', 'apple'])
df1.reindex(['cherry', 'banana', 'mango', 'apple'])

# 2. emp.csv 파일을 읽고
df2 = pd.read_csv('emp.csv')

# 1) 이름컬럼 값을 모두 소문자로 변경
'ALLEN'.lower()
df2.ENAME.lower()    # 벡터연산 불가
df2.ENAME = df2.ENAME.map(lambda x : x.lower())

# 2) 아래 인상 규칙대로 인상된 연봉을 계산후 new_sal 컬럼에 추가
# (컬럼 추가 방식 : df1['new_sal'] = values)
# 10번부서는 10%, 20번 부서는 15%, 30번 부서는 20%
df2.dtypes

# 1) for문
for i,j in zip(df2.DEPTNO, df2.SAL) :
    print('부서번호 : %s, 연봉 : %s' % (i,j))

vsal = []

for i,j in zip(df2.DEPTNO, df2.SAL) :
    if i == 10 :
        vsal.append(j * 1.1)
    elif i == 20 :
        vsal.append(j * 1.15)
    else :
        vsal.append(j * 1.2)

df2['new_sal'] = vsal

# 2) np.where
vsal2 = np.where(df2.DEPTNO == 10, df2.SAL * 1.1, 
                 np.where(df2.DEPTNO == 20, df2.SAL * 1.15, df2.SAL * 1.2))

df2['new_sal2'] = vsal2

# 3) mapping 처리****
f1 = lambda x : '인사부' if x == 10 else '총무부'
df2.DEPTNO.map(f1)

def f2(x,y) :
    if x == 10 :
        return(y * 1.1)
    elif x == 20 :
        return(y * 1.15)
    else :
        return(y * 1.2)
    
f2(10,800)               # 스칼라 전달 가능
f2(df2.DEPTNO, df2.SAL)  # 벡터연산 불가

df2.DEPTNO.map(f2, df2.SAL)              # map 메서드는 추가 인자 전달 불가
l1 = list(map(f2, df2.DEPTNO, df2.SAL))  # map 함수는 추가 인자 전달 가능

df2['new_sal3'] = l1


# [ 위 for문 사용시 주의 ]
# df2['new_sal2'][0] = 10   # 없는 키에 값 직접 할당 불가(R에서는 가능한 표현식)

# for i in range(0,14) :    # df2['new_sal2'] 가 없어서 불가
#     if df2.loc[i,'DEPTNO'] == 10 :
#         df2['new_sal2'][i] = df2.loc[i,'SAL'] * 1.1
#     elif i == 20 :
#         df2['new_sal2'][i] = df2.loc[i,'SAL'] * 1.15
#     else :
#         df2['new_sal2'][i] = df2.loc[i,'SAL'] * 1.2

# # -- out of range 상황 해결하기위해 리스트의 전체 범위에 값 임의 할당
# l1=list(np.arange(1,15))

# for i in range(0,14) :    # df2['new_sal2'] 가 없어서 불가
#     if df2.loc[i,'DEPTNO'] == 10 :
#         l1[i] = df2.loc[i,'SAL'] * 1.1
#     elif i == 20 :
#         l1[i] = df2.loc[i,'SAL'] * 1.15
#     else :
#         l1[i] = df2.loc[i,'SAL'] * 1.2

# 3) comm이 없는 직원은 100 부여
# 1) np.where
np.where(pd.isnull(df2.COMM), 100, df2.COMM)

# 2) map
df2.COMM.map(lambda x : 100 if pd.isnull(x) else x)

########## 여기까지는 복습입니다. ##########

# DataFrame
# 1. 생성
# 2. 색인
df1.iloc
df1.loc

df1.iloc[-1, :]   # 제외의 의미X
df1.iloc[1:, :]   # 첫번째 행 제외 X

df1.drop('c', axis=0)  # 행 제거
df1.drop('C', axis=1)  # 컬럼 제거
df1.drop(0, axis=1)    # 에러, 위치값 전달 불가

df2 = DataFrame(np.arange(1,5).reshape(2,2))
df2.drop(0, axis=0)    # 0이 index 이름이므로 가능

# drop 메서드
# - 특정 행, 컬럼을 제외
# - 원본 수정 X
# - axis 인자로 행(0), 컬럼(1) 방향 지정
# - 이름으로만 가능(positional index값 전달 불가)


# 3. 기본 메서드
df1.index
df1.columns
df1.values         # key값 제외한 순수 데이터

df1.index.name     # index의 이름
df1.columns.name   # index의 이름

pro.index.name = 'rownum'
pro.columns.name = 'colname'

df1.dtypes

# 4. index 수정
df1 = DataFrame(np.arange(1,17).reshape(4,4))
df1.index = ['a','b','c','d']
df1.columns = ['A','B','C','D']

# index object의 일부 수정
df1.columns[-1] = 'col4'  # index object의 일부 수정 불가

# 1) 다른 객체 생성 및 변경 후 index object 덮어쓰기
v1 = df1.columns
v2 = df1.columns.values

v1[-1] = 'col4'           # 같은 에러 발생
v2[-1] = 'col4'           # 가능
df1.columns = v2

# 2) rename 메서드 활용(index object 변경 수행) ****
df1.rename({'C':'col3'}, axis=0)  # 변경X, index(행) 이름 수정
df1.rename({'C':'col3'}, axis=1)  # 변경O, column 이름 수정


# [ 연습 문제 ]
run profile1
pd.read_csv('professor.csv', encoding='cp949')
pro = pd.read_csv('professor.csv', engine='python')

# 1) 홈페이지 주소가 있는 경우 그대로, 없으면 
#    http://www.kic.com//email_id

# 1 - mapping 처리
f1 = lambda x,y : 'http://www.kic.com/' + y.split('@')[0] 
                   if pd.isnull(x) else x  

list(map(f1, pro.HPAGE, pro.EMAIL))

# 2 - 반복문 처리
l1 = []
for i,j in zip(pro.EMAIL, pro.HPAGE) :
    if pd.isnull(j) :
        l1.append('http://www.kic.com//%s' % (i[0:i.find('@')]))
    else :
        l1.append(j)

# 2) avg_sal 컬럼에 각 행마다 각 행의 부서번호를 확인후,
#    같은 부서의 평균 avg 값을 삽입
pro.loc[pro.DEPTNO == 101, 'PAY'].mean()
pro.loc[pro.DEPTNO == 102, 'PAY'].mean()

# 1 - mapping 처리
f2 = lambda x : pro.loc[pro.DEPTNO == x, 'PAY'].mean()

pro['avg_sal'] = pro.DEPTNO.map(f2)

# 2 - 반복문 처리
l2 = []
for i in pro.DEPTNO :
    l2.append(np.mean(pro.loc[pro.DEPTNO == i, 'PAY']))
pro['avg_sal'] = l2 

# 5. 구조 수정
df1 = DataFrame(np.arange(1,9).reshape(2,4))
df2 = DataFrame(np.arange(9,21).reshape(3,4))

# 1) row 추가(rbind)
df1.append([10,11,12,13])  # key error(10~13의 값이 각 컬럼별로 삽입 X)
df1.append(df2)            # 행 추가시 key(컬럼)가 같은 값이 추가

df1.append(df2, ignore_index=True) # 행 추가시 기존 index값 무시

# 2) column 추가(cbind)
df1['4'] = [10,20]


# 6. 산술 연산
# - 같은 index, 같은 column끼리 매칭시켜 연산처리
# - 매칭되지 않는 index의 연산결과는 NA
# - 산술연산 메서드(add,sub,mul,div)는 NA로 리턴되는 현상 방지

df1 = df1.drop('4', axis=1)

df1.columns = ['a','b','c','d']
df2.columns = ['a','b','c','d']

df1 + df2      # 서로 다른 크기의 데이터 프레임 산술연산 가능(key끼리 매칭)

df2.add(df1, fill_value=0)
df2.mul(df1, fill_value=1)

# 사칙연산시 fill_value값 전달
# 1 + NA, NA가 0으로 수정
# 1 * NA, NA가 1로 수정
# 1 / NA, NA가 1로 수정
# 1 - NA, NA가 0으로 수정

# =============================================================================
# [ 참고 - df1 + df2 처리 방식 ]
#
# 1. df1을 df2처럼 변경(index가 0,1,2값을 갖도록)
df1.reindex(df2.index)
# 
# 2. 위 대상과 df2를 연산처리
df1.reindex(df2.index) + df2
# 
# =============================================================================

# [ 참고 - numpy와 pandas의 산술연산 메서드 비교 ]
a1 = np.array([4,1,10,NA])
a1.mean()                   # numpy의 mean 호출
np.mean(a1)
np.mean?

s1 = Series(a1)
s1.mean()                   # pandas의 mean 호출(NA 무시가 기본)
s1.mean(skipna=False)       # numpy에서처럼 NA 무시 X

s1.mean?

# =============================================================================
# 참고 in SQL
# 
# select avg(comm),           # comm이 있는 직원만 평균 계산
#        sum(comm)/count(*),  # 전체 직원 평균 계산
#        avg(nvl(comm,0))     # 전체 직원 평균 계산
#   from emp;
# 
# =============================================================================


# [ 연습 문제 ]
# 1. emp.csv 파일을 읽고
emp = pd.read_csv('emp.csv')

# 1) index값을 사원번호로 설정
emp.index = emp.EMPNO
emp = emp.drop('EMPNO', axis=1)

# 2) 컬럼이름을 모두 소문자로 변경
emp.columns = emp.columns.map(lambda x : x.lower())

# 3) 전체 직원의 comm의 평균(comm이 있는 직원의 평균, 전체 평균)
emp.comm.mean()   # 4명의 평균

# -- NA를 0으로 수정
np.where(pd.isnull(emp.comm),0,emp.comm)
emp.comm.map(lambda x : 0 if pd.isnull(x) else x)
emp.comm[pd.isnull(emp.comm)] = 0

emp.comm.mean()   # 14명의 평균

# 4) 7902 행 제거
emp = emp.drop(7902, axis=0)

# 5) hiredate 컬럼을 hdate로 변경
emp = emp.rename({'hiredate':'hdate'}, axis=1)



# 7. 정렬
emp.sort_values(by,          # 정렬할 컬럼
                axis,        # 정렬 방향
                ascending,   # 오름차순 정렬 여부
                inplace,     # 원본 수정 여부
                na_position) # NA 배치 순서


emp.sort_values(by='ename')
emp.sort_values(by='ename', ascending=False)
emp.sort_values(by=['deptno', 'sal'], ascending=[True, False])
emp.sort_values(by=['deptno', 'sal'], ascending=[True, False], inplace=True)


# 8. reindex 기능
df1 - df2
df1.sub(df2)
df1.sub(df2, fill_value=0)

df1**df2
df1 = df1.reindex(df2.index, fill_value=1)
df1**df2

