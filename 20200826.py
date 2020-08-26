# array 생성 옵션
import numpy as np

# 1) type 지정
l1 = [[1,2,3],[4,5,6]]
np.array(l1, dtype=float)

# 2) order(배열순서) 지정
a1 = np.arange(1,10)

a1.reshape(3,3,order='C')  # 행 우선순위 배치(기본값)
a1.reshape(3,3,order='F')  # 컬럼 우선순위 배치

# [ 연습 문제 ]
# disease.txt 파일을 읽고(컬럼명 생략) 맨 마지막 컬럼 데이터를
# 소수점 둘째자리까지 표현 후 다시 새로운 파일에 저장
from my_func import *
l1 = f_read_txt('disease.txt', sep='\t', fmt=str)
l2 = l1[1:]
a1 = np.array(l2)
a1.ndim

# 마지막 컬럼 선택
a1[:,-1]

# 'NA' => '0'
a1[:,-1][-1] = '0'
a1[:,-1].replace('NA','0')  # array에 적용 불가

a2 = [x.replace('NA','0') for x in a1[:,-1]]

# 소수점 둘째자리 표현
'%.2f' % int(a2[0])

a1[:,-1] = ['%.2f' % int(x.replace('NA','0')) for x in a1[:,-1]]

f_write_txt(a1,'disease2.txt')


# 형 변환 함수/메서드
# 1. 함수 : int, float, str
# - 벡터연산 불가

float(a1[:,-1])                           # 불가
[float(x) for x in a1[:,-1]]              # 가능 
list(map(lambda x : float(x), a1[:,-1]))  # 가능

# 2. 메서드 : astype
# - array, Series, Dataframe에 적용 가능
# - 벡터연산 가능

a1[:,-1].astype('float')


# [ 연습 문제 : 다음의 값의 10% 인상된 값 출력 ]
arr2 = np.array(['1,100','2,200','3,300'])

[ x.replace(',','').astype('int') for x in arr2 ]  # 불가
[ int(x.replace(',','')) for x in arr2 ]           # 가능

np.array([ x.replace(',','') for x in arr2 ]).astype('int') # 가능

# numpy의 산술연산 함수 및 메서드
a1 = np.array([1,2,3])
a2 = np.arange(1,10).reshape(3,3)

np.sum(a1)      # 총 합
a2.sum()
a2.sum(axis=0)  # 행별(서로다른 행끼리, 세로방향)
a2.sum(axis=1)  # 열별(서로다른 열끼리, 가로방향)

np.mean(a1)     # 평균
a2.mean()

np.var(a1)      # 분산
a2.var()

np.std(a1)      # 표준편차
a2.std()        # 

a2.min(0)
a2.max(1)

a2.cumsum(0)     # 누적합
a2.cumprod(1)    # 누적곱

a1.argmin()      # 최소값 위치 리턴(whichmin in R)
a1.argmax()      # 최대값 위치 리턴(whichmax in R)

a2.argmin(0)


# 논리 연산 메서드
(a2 > 5).sum()  # 참의 개수
(a2 > 5).any()  # 하나라도 참인지 여부
(a2 > 5).all()  # 전체가 참인지 여부

# [ 연습 문제 ]
# 다음의 구조를 갖는 array를 생성하자. 
# 1   500     5  
# 2   200     2  
# 3   200     7  
# 4    50     9 

arr5 = np.array([[1,500,5],[2,200,2],[3,200,7],[4,50,9]])

# 1) 위의 배열에서 두번째 컬럼 값이 300이상인 행 선택
arr5[arr5[:,1] >= 300, :]

# 2) 세번째 컬럼 값이 최대값인 행 선택
arr5[arr5[:,2] == arr5[:,2].max(), :] 
arr5[arr5[:,2].argmax(), :]

# =============================================================================
# [ 참고 : numpy와 pandas의 분산의 차이 ]
#
# from pandas import Series
# 
# a1.var()         # 0.67
# s1 = Series(a1)
# s1.var()         # 1
# 
# s1.var(ddof=0)   # 0.67
#    
# ((1-2)**2 + (2-2)**2 + (3-2)**2)/3  # 0.67
# ((1-2)**2 + (2-2)**2 + (3-2)**2)/2  # 1
# 
# sum((a1 - a1.mean())**2)/3
# sum((a1 - a1.mean())**2)/2
# 
# sum((x - xbar)**2)/n         # numpy의 분산 계산 식
# sum((x - xbar)**2)/(n-1)     # pandas의 분산 계산 식
# sum((x - xbar)**2)/(n-ddof)  # pandas의 분산 계산 식
# =============================================================================


# [ 연습 문제 ]
# 다음의 배열에서 행별, 열별 분산을 구하여라
a1 = np.array([[1,5,9],[2,8,3],[6,7,1]])

a1.var(0)
a1.var(1)

# 행별, 열별 평균 구하기
a1.mean(0) 
a1.mean(1)

# 1) 행별 연산
a1 - a1.mean(0)                  # 행별(서로다른 행끼리,세로방향) 편차
(a1 - a1.mean(0))**2             # 행별(서로다른 행끼리,세로방향) 편차 제곱
np.sum((a1 - a1.mean(0))**2, 0)     # 행별 편차 제곱의 합
np.sum((a1 - a1.mean(0))**2, 0) / 3 # 행별 분산
a1.var(0)

1-3, 5-6.7, 9-4.3
2-3, 8-6.7, 3-4.3
6-3, 7-6.7, 1-4.3

=========
3 6.7 4.3
 
# 2) 열별 연산
np.sum((a1 - a1.mean(1))**2, 1) / 3                # 잘못된 연산 결과
np.sum(((a1 - a1.mean(1).reshape(3,1))**2),1) / 3  # 올바른 연산 결과
a1.var(1)

1-5  , 5-5  , 9-5    | 5
2-4.3, 8-4.3, 3-4.3  | 4.3
6-4.7, 7-4.7, 1-4.7  | 4.67

((1-5)**2 + (5-5)**2 + (9-5)**2) / 3


# array의 deep copy
arr1 = np.array([1,2,3,4])
arr2 = arr1         # 원본 배열의 뷰 생성
arr3 = arr1[0:3]    # 원본 배열의 뷰 생성
arr4 = arr1[:]      # 원본 배열의 뷰 생성
arr5 = arr1.copy()  # 서로 다른 객체 생성

arr1[0] = 10
arr1[1] = 20
arr1[2] = 30

arr2[0]       # arr2도 변경, deep copy 발생 X
arr3[0]       # arr3도 변경, deep copy 발생 X
arr4[1]       # arr4도 변경
arr5[2]       # arr5는 변경되지 X, deep copy 발생



# np.where
# - R의 ifelse 구문과 비슷
# - np.where(조건, 참리턴, 거짓리턴)
# - 거짓리턴 생략 불가

np.where(a1 > 5, 'A', 'B')
np.where(a1 > 5, 'A')        # 거짓리턴 생략 불가

# [ 연습 문제 ]
# emp.csv파일의 부서번호를 사용, 부서이름 출력
# 10이면 인사부 20이면 총무부 30이면 재무부

a_emp = np.array(f_read_txt('emp.csv',sep=',',fmt=str))
vdeptno = a_emp[1:,-1]

# 1) mapping + 함수
def f1(x) :
    if x == '10' :
        return '인사부'
    elif x == '20' :
        return '총무부'
    else :
        return '재무부'

f1('10')     # 스칼라에 대해 조건 치환 가능
f1(vdeptno)  # 에러, if문을 사용한 조건 치환은 벡터 연산 불가

list(map(f1,vdeptno))

# 2) np.where
np.where(vdeptno == '10', '인사부', 
                          np.where(vdeptno == '20', '총무부', '재무부'))


# 전치 메서드
# 1. T
# - 행, 열 전치**
# - 3차원 이상은 역 전치

# 2. swapaxes
# - 두 축을 전달 받아 전치

# 3. transpose
# - 여러 축 동시 전치 가능

arr1 = np.arange(1,10).reshape(3,3)
arr2 = np.arange(1,25).reshape(2,3,4)

# 2차원 배열의 행, 열 전치
arr1     # 3X4
arr1.T

arr1.swapaxes(0,1)
arr1.swapaxes(1,0)

arr1.transpose(1,0)
arr1.transpose(0,1)

arr2.T   # 4X3X2 (원래 배열은 2층,3행,4열)

# 층과 열 전치
arr2.swapaxes(0,2).shape    # 4 X 3 X 2     
arr2.transpose(층,행,열)
arr2.transpose(2,1,0).shape # 4 X 3 X 2    


# =============================================================================
# [ 참고 - 축 번호 비교 ]
#
# in python
# 
#     행 열       층 행 열
#     0  1        0  1 2
# 
# in R     
#  
#     행 열       행 열  층
#     1  2        1  2  3
# =============================================================================


# 정렬
l1 = [2,1,5,7,3]
l1.sort()                     # 원본 수정
l1.sort(reverse=True)         # 역순 정렬

arr6 = np.array([2,3,6,9,1])

arr6.sort()                   # array 적용 가능한 정렬 메서드
arr6.sort(order=True)         # 오름차순만 가능, 내림차순 전달 불가

arr5.sort(0) ; arr5           # 정렬 방향 지정 가능


# numpy의 집합 연산자
# 1. union1d : 합집합
# 2. intersect1d : 교집합
# 3. setdiff1d : 차집합
# 4. in1d : 포함 연산자
# 5. setxor1d : 대칭 차집합
# 6. unique : unique value

a1
a22 = a2.reshape(1,-1)

np.union1d(a1, a2)      
np.intersect1d(a1, a2)
np.setdiff1d(a2, a1)
np.setxor1d(a1, a2)     #(a1 - a2) U (a2 - a1)

'x' in 'cxb'
1 in [1,2,3]

np.in1d(a1,a2)          # a1[0] in a2, a1[1] in a2 , a1[2] in a2
[ x in a2 for x in a1]  # a1[0] in a2, a1[1] in a2 , a1[2] in a2

np.unique([1,1,1,2])
set([1,1,1,2])

np.unique(np.array([1,1,1,2]))


# [ 연습 문제 ]
# 1~25의 값을 갖는 5X5 배열을 생성후 2의 배수와 3의 배수를 추출
arr6 = np.arange(1,26).reshape(5,5)

arr7 = arr6[arr6 % 2 == 0]
arr8 = arr6[arr6 % 3 == 0]

np.union1d(arr7, arr8)     



# NA
# - NA   : 자리수 차지, 결측치(잘못 입력된 값)
# - NULL : 자리수 차지 X

from numpy import nan as NA

[1,2,NA]      # 숫자와 NA 함께 표현 가능
['1','2',NA]  # 문자와 NA 함께 표현 가능

# 정수형 array에 float형 nan 삽입 불가
a1.dtype
a1[0] = NA    # cannot convert float NaN to integer

a2 = a1.astype('float')
a2[0] = NA

# nan 확인 함수
a2[np.isnan(a2)]

np.isnan(a2).sum()
np.isnan(a2).any()


# numpy에서의 파일 입출력 함수
np.loadtxt(fname,        # 불러올 파일 이름
           dtype,        # 저장할 데이터 포맷
           delimiter,    # 구분기호
           skiprows,     # skip할 행 개수
           usecols,      # 불러올 컬럼
           encoding)     # encodnig

np.loadtxt('file1.txt', dtype='int', delimiter=',')
np.loadtxt('file3.txt', dtype='int', delimiter=',', skiprows=1)
np.loadtxt('file3.txt', dtype='int', delimiter=',', 
                        skiprows=1, usecols=[0,2])


np.savetxt(fname,        # 저장할 파일명
           X,            # 저장할 객체
           fmt,          # 포맷
           delimiter)    # 분리구분기호


np.savetxt('arr_test1.txt', arr5, delimiter=';', fmt='%.2f')

