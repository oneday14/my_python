# math 모듈 내 함수
import math
dir(math)       # 모듈 내 함수 목록 확인

math.sqrt(9)    # 제곱근

2**3            # 거듭제곱 연산자
math.pow(2,3)   # 거듭제곱 함수

# 문자열 관련 표현식
# 1. 문자열 색인(추출)
v1='abcde'
v1[1]
v1[0:3]

# 2. 다중 라인 문자열 생성
v2 = 'abcde
12345'

vsql='select *
        from emp'

vsql='''select *
        from emp'''

vsql="""select *
        from emp"""


# 3. 문자열 메서드
s1='abcde'
s2='AbCde'
l1=['abc','ABC']        

# 1) 대소치환
s1.upper()
s2.lower()
s1.title()  # camel 표기법

# 2) startswith : 문자열의 시작 여부 확인
s1.startswith(prefix, # 시작값 확인문자
              start,  # 검사 시작 위치(생략가능)
              end)    # 검사 끝 위치(생략가능)

s1.startswith('A')  
      
s1[2] == 'c'
s1.startswith('c',2)

# 3) endswith : 문자열의 끝 여부 확인
s1.endswith('e')

l1.startswith('a')   # 리스트에 문자열 메서드 적용 불가(적용함수 사용)

# 4) strip : 공백 또는 문자 제거
' abcd '.rstrip()    # 오른쪽 공백 제거
' abcd '.lstrip()    # 왼쪽 공백 제거
' abcd '.strip()     # 양쪽 공백 제거

'aaaabcade'.lstrip('a') # 왼쪽에서 'a' 제거

# 5) replace : 치환
'abcde'.replace('a','A')  # 치환
'abcde'.replace('c','')   # 제거

# 6) split : 분리
'a;b;c'.split(';')[1]

# 7) find : 특정문자의 위치값 리턴
'abcde'.find('a')   # 'a'의 위치 리턴
'abcde'.find('A')   # 없으면 -1 리턴

'abcde fg 1234'.find('1234')

# 8) count : 포함 횟수 리턴
'abcabcaa'.count('a')

# 9) 형(type) 확인
'abd'.isalpha()    # 문자 구성 여부
'abd'.isnumeric()  # 숫자 구성 여부
'abd'.isalnum()    # 문자/숫자 구성 여부

# 10) format : 포맷 변경
'{0:.2f}'.format(10)


# 연습문제
ename = 'smith'
tel = '02)345-7839'
jumin = '901223-2223928'
vid = 'abc1234!'

# 1) ename의 두번째 글자가 m으로 시작하는 지 여부
ename[1] == 'm'
ename.startswith('m', 1)

# 2) tel에서 국번(345) 출력
vno1 = tel.find(')')
vno2 = tel.find('-')

tel[vno1+1:vno2]

tel.split(')')[1].split('-')[0]

# 3) jumin에서 여자인지 여부 출력
jumin[7] == '2'

# 4) vid에서 '!'가 포함되어 있는지 여부 출력
vid.find('!') != -1


# 4. 문자열 결합
'a' + 'b'

# 5. 패턴확인(포함여부)
'abcde'.find('c')

'c' in 'abcde'

# 6. 문자열 길이 확인
len('abcde')

# =============================================================================
# [ 참고 - sql, R에서의 문자열 결합 ]
# 'a' || 'b'               # in sql
# stringr::str_c('a','b')  # in R
# paste('a','b',sep='')    # in R
# =============================================================================


# [ 참고 : 함수와 메서드의 차이 ]
func1(x,y,z) = x + y + z

func1(data,x,y)
data.func1(x,y)


from pandas import Series
s1 = Series([1,2,3,4])

s1.map?

# [ 연습 문제 - 지폐 계산 프로그램 ]
money = int(input('지폐로 교환할 돈을 얼마? : '))

q50 = money // 50000    # %/% in R
money = money % 50000   # %%  in R

q10 = money // 10000    
money = money % 10000 

q5 = money // 5000    
money = money % 5000 

q1 = money // 1000    
money = money % 1000 

print('50000원짜리 ==> %2d장' % q50)
print('10000원짜리 ==> %2d장' % q10)
print(' 5000원짜리 ==> %2d장' % q5)
print(' 1000원짜리 ==> %2d장' % q1)

print('나머지 금액 ==> %d원' % money)


# 논리연산자
v1 = 100

(v1 > 50) and (v1 < 150)
(v1 > 50) & (v1 < 150)

(v1 > 50) or (v1 < 150)
(v1 > 50) | (v1 < 150)

not(v1 > 50)              # in R !(v1 > 50)


# 리스트 조건 전달
l1 = [1,2,3,4,5]
l1 > 3            # 리스트와 정수의 대소 비교 불가
l1 == 1           # 같다, 같지 않다는 벡터 연산 불가


# =============================================================================
# lambda
# - 축약형 사용자 정의 함수 생성 문법
# - 간단한 return만 가능
# - 복잡한 프로그래밍 처리 불가
# - 함수명 = lambda input : output
# =============================================================================
f1 = lambda x : x + 1    # y = x + 1
f1(5)

f2 = lambda x, y : x + y    
f2(1,10)

f3 = lambda x, y=0 : x + y  # 뒤의 input value는 디폴트 값 선언 가능    
f3 = lambda x=0, y : x + y  # 앞의 input value만 디폴트 값 선언 불가    
f3(1)

# =============================================================================
# map
# - 1차원 적용함수 : 데이터 셋의 함수의 반복 적용 가능
# - 결과 출력시 list 함수 사용 필요
# - 함수의 추가 인자 전달 가능
# - map(func, *iterables)
# =============================================================================
l1 + 1            # 벡터연산 불가
list(map(f1, l1)) # mapping 처리 가능

# 예제) 다음 두 리스트의 각 원소의 합을 출력
l1=[1,2,3,4,5]
l2=[10,20,30,40,50]

l1 + l2               # 리스트 연산 불가, 확장

f2 = lambda x, y : x + y
list(map(f2,l1,l2))


# 예제) 다음 리스트의 각 원소를 대문자로 변경
l3=['a','b','ab']
'a'.upper()        # 문자열에 적용 가능
l3.upper()         # 리스트에 적용 불가 

f3 = lambda x : x.upper()

list(map(f3, l3))


# [ 연습 문제 ]
# 다음의 리스트를 생성
L1 = [1,2,3,4]
L2 = [10,20,30,40] 
L4 = ['서울','부산','대전','전주']
L5 = ['abc@naver.com','a123@hanmail.net']

# 1. L2의 L1승 출력, 10^1, 20^2, 30^3, 40^4
L2**L1

f1 = lambda x, y : x**y
list(map(f1,l2,l1))

# 2. L4의 값에 "시"를 붙여 출력
'서울' + '시'
L4 + '시'

f2 = lambda x : x + '시'
list(map(f2,L4))

# 3. L5에서 이메일 아이디만 출력
f3 = lambda x : x[0:x.find('@')]
list(map(f3,L5))

'abc@naver.com'.split('@')[0]

f4 = lambda x : x.split('@')[0]
list(map(f4,L5))

# [ 예제 ]
# 위 문제에서 split 메서드를 사용하여 전달되는 분리구분기호로 분리,
# 전달되는 위치값에 해당되는 원소 추출

f5 = lambda x, y, z : x.split(y)[z]
f6 = lambda x, y=';', z=0 : x.split(y)[z]

l1 = ['a;b;c', 'A;B;C']

f5('a;b;c',';',1)
f5(l1,';',1)            # 불가

list(map(f5,l1,';',1))            # 불가
list(map(f5,l1,[';',';'],[0,1]))  # 가능


# =============================================================================
# [ 참고 - 함수의 인자 전달 방식 in R ]
#
# v1 <- c('a;b;c', 'A;B;C')
# 
# f1 <- function(x,y,z) {
#     str_split(x,y)[[1]][z]
# }
# 
# sapply(v1, f1, ';', 1)
# =============================================================================


# [ 예제 ]
# 다음의 리스트의 각 원소가 a로 시작하는지 여부 확인
l1 = ['abc','ABC','bcd']
l1.startswith('a')         # 불가

list(map(lambda x : x.startswith('a'), l1))


l1[[True, False, False]]          # 리스트는 리스트색인 불가(조건색인 불가)
Series(l1)[[True, False, False]]  # 시리즈는 리스트색인 가능(조건색인 가능)



# 리스트
# 1. 생성
l1 = [1,2,3,4]
l2 = [1,2,3,4,5,6]

# 2. 색인
l1[0]
l1[0:1]
l1[[0,2]]
l1[-1]
l1[::2]    # 시작값:끝값+1:증가값

# 3. 연산
l1 + 1
list(map(lambda x : x + 1, l1))

# 4. 확장
l1 + l2
l1.append(5)   # 원소를 추가하여 원본 객체를 즉시 수정

l2 + [7]       # 추가할 값을 리스트로 전달하여 + 연산 시 추가
l2.extend([7]) # 리스트를 추가하여 원본 객체를 즉시 수정

# 5. 수정
l1[0] = 10  ; l1
l1[1] = [20,30]  ; l1
l1[2:4] = [300,400]  ; l1

l2[7] = 8   # out ou range 에러 발생, 정의되지 않은 position에 값 할당 불가

# 6. 삭제
del(l1[1])  # 값 삭제 후 원본 수정
del(l1)     # 객체 자체 삭제 가능
l2 = []     # 객체는 유지, 원소만 전체 삭제


# =============================================================================
# [ 참고 - R에서의 벡터 확장 방식 ]
# v1 <- c()
# for ... {
#   v1[i] <- ...
#   v1 <- v(v1, ...)
# }
# =============================================================================

