# 1. 문자열, 찾을 문자열, 바꿀 문자열을 입력 받아 변경한 결과를 아래와 같이 출력
# 전 :
# 후 :

str1 = input('치환할 원본 문자열 : ')
str2 = input('찾을 문자열 : ')
str3 = input('바꿀 문자열 : ')

re1 = str1.replace(str2, str3)

print('전 : %s' % str1) 
print('후 : %s' % re1)


# 2. 이메일 주소를 입력받고 다음과 같이 출력
# 아이디 : a1234
# 메일엔진 : naver

vemail = input('이메일 주소를 입력하세요 : ')

vid = vemail.split('@')[0]
vaddr = vemail.split('@')[1].split('.')[0]

print('아이디 : %s' % vid)
print('메일엔진 : %s' % vaddr)


# 3. 2번을 활용하여 다음과 같은 홈페이지 주소 출력
# http://kic.com/a1234

vhpage = 'http://kic.com/' + vid

print('홈페이지 주소 : %s' % vhpage)

# 4. num1='12,000' 의 값을 생성 후, 33으로 나눈 값을 소숫점 둘째짜리까지 표현
num1='12,000'

round(int(num1.replace(',','')) / 33, 2)

import math
math.trunc(26.987, 2)  # 자리수 전달 불가

# 5. 다음의 리스트 생성 후 연산
ename = ['smith','allen','king'] 
jumin = ['8812111223928','8905042323343','90050612343432']
tel=['02)345-4958','031)334-0948','055)394-9050','063)473-3853']
vid=['2007(1)','2007(2)','2007(3)','2007(4)']

# 1) ename에서 i를 포함하는지 여부 확인
'i' in 'smith'
'i' in ename    # 벡터연산 불가, 포함 여부 확인 불가(정확한 일치)

f1 = lambda x : 'i' in x
list(map(f1, ename))

# 2) jumin에서 성별 숫자 출력
'8812111223928'[6]

f2 = lambda x : x[6]
list(map(f2, jumin))

# 3) ename에서 smith 또는 allen 인지 여부 출력 [True,True,False]
'smith' in ['smith', 'allen']
('smith' == 'smith') or ('smith' == 'allen')

f3 = lambda x : x in ['smith', 'allen']
f3 = lambda x : (x == 'smith') or (x == 'allen')

list(map(f3, ename))

# in oracle : ename in ('smith', 'allen')

# 4) tel에서 다음과 같이 국번 XXX 치환 (02)345-4958 => 02)XXX-4958)
'02)345-4958'.split(')')[1].split('-')[0]
'02)345-4958'.replace('02)345-4958'.split(')')[1].split('-')[0],'XXX')

f4 = lambda x : x.replace(x.split(')')[1].split('-')[0],'XXX')
list(map(f4, tel))

# 5) vid 에서 각각 년도와 분기를 따로 저장
'2007(1)'[:4]
'2007(1)'[5]

f5 = lambda x : x[:4]
f6 = lambda x : x[5]
f7 = lambda x : [x[:4], x[5]]

vyear = list(map(f5, vid))
vqt = list(map(f6, vid))

list(map(f7, vid))

########## 여기까지는 복습입니다. ##########

# 리스트 메서드
l1=[1,2,3,4]

# 1. append : 리스트 맨 끝에 원소 추가
l1.append(5) ; l1

# 2. extend : 리스트 맨 끝에 원소 추가
l1 + [6]
l1.extend([6]) ; l1

# 3. insert : 특정 위치에 원소 추가 가능
l1.insert(0,0) ; l1

# 4. remove : 특정 원소 제거
l2 = ['a','b','c','d']
del(l2[0]) ; l2
l2.remove('c') ; l2

# 5. pop : 맨 끝 원소/특정 위치 제거
l1.pop() ; l1   # 맨 끝 원소 제거
l1.pop(0) ; l1  # 전달한 위치값 제거 

# 6. index : 리스트 내 특정 원소의 위치 리턴
l2.index('b')   # 있으면 위치 리턴
l2.index('B')   # 없으면 에러 처리

# 7. count : 리스트 내 특정 원소의 포함 횟수
l4 = [1,1,1,2,3]
l4.count(1)
'aaabcd'.count('a')

# 8. sort : 정렬(원본 수정)
l5=[4,1,6,3,8]
l5.sort() ; l5
l5.sort(reverse=True) ; l5

# 9. reverse : 거꾸로 출력(원본 수정)
l4[::-1]
l4.reverse() ; l4

# 10. len : 문자열 크기 또는 리스트의 원소의 개수
len(l5)


# 2차원 리스트의 표현
- 리스트의 중첩 구조를 사용하여 마치 2차원인것처럼 출력 가능
- 반복문 필요

l1 = [[1,2,3],[4,5,6],[7,8,9]]
l[5,5]    # 2차원이 아니므로 2차원 색인 형태 불가
l1[1][1]  # 순차적 색인 필요

# for문
for i in range(0, 10) :
    반복문

for i in range(0,5) :
    for j in range(0,10) :
        내부 for문 반복문
        
    외부 for문 반복문    

for i in range(0,5) :
    for j in range(0,10) :
        내부 for문 반복문
    외부 for문 반복문  

# =============================================================================
# in R
# for (i in 1:10) {
#   반복문
# } 
# =============================================================================

# [ 예제 - 1~10까지 출력 ]
for i in range(1,11) :
    print(i)

for i in range(1,11) :
    print(i, end=';')
    
# [ 예제 - 1~10까지 홀수만 출력 ]
for i in range(1,11,2) :     # 1:11:2
    print(i)


# [ 연습 문제 ]
# 다음의 리스트에서 지역번호를 추출(for문 사용)
tel = ['02)345-9384','031)3983-3438','032)348-3938']

vtel = []

for i in tel :
    vno = i.find(')')
    vtel.append(i[:vno])

vtel


# [ 연습 문제 ]
# 사용자로부터 시작값, 끝값, 증가값을 입력받은 후
# 시작값부터 끝값 사이의 해당 증가값 대상의 총 합을 계산 후 출력 (for문 사용)

no1 = int(input('시작값 : '))
no2 = int(input('끝값 : '))
no3 = int(input('증가값 : '))

for i in range(no1,no2+1,no3) :
    print(i)

vsum = 0
for i in range(no1,no2+1,no3) :
    vsum = vsum + i

vsum

print('%d에서 %d까지 %d씩 증가값의 총 합 : %d' % (no1, no2, no3, vsum))


# 중첩 for문 
l1 = [[1,2,3],[4,5,6],[7,8,9]]
l2 = [[1,2,3],[4,5,6],[7,8,9,10]]

for i in l2 :
    for j in i :
        print(j, end=' ')
    print()
    
    
# i          j
# [1,2,3]    1
#            2
#            3
           
# [4,5,6]    4
#            5
#            6
           
# [7,8,9,10] 7
#            8
#            9
#            10
           
    
for i in range(0,3) :
    for j in range(0,3) :
        print(l1[i][j], end=' ')
    print()                 
    
# l1 = [[1,2,3],[4,5,6],[7,8,9]]
# i     j
# 0     0   l1[0][0] = 1
#       1   l1[0][1] = 2
#       2   l1[0][2] = 3
      
# 1     0   l1[1][0] = 4
#       1   l1[1][1] = 5
#       2   l1[1][2] = 6

for i in range(0,len(l2)) :
    for j in range(0,len(l2[i])) :
        print(l2[i][j], end=' ')
    print()  

# l2 = [[1,2,3],[4,5,6],[7,8,9,10]]

f_write_txt(l2,sep=';')


# [ 연습 문제 ]
# 원본문자열과 찾을 문자열, 바꿀 문자열을 차례대로 입력받고
# translate 기능으로 각 글자마다의 치환 후 결과를 다음과 같이 출력
# abcdeba, 'abc', '123' => 123de21 
# (스칼라 테스트 후 리스트 확장)

# 전 : 
# 후 :

str1 = input('원본 문자열 입력 : ')
vold = input('찾을 문자열 입력 : ')
vnew = input('바꿀 문자열 입력 : ')

str2 = str1

for i in range(0,len(vold)) :
    str2 = str2.replace(vold[i], vnew[i])

print('전 : %s' % str1)
print('후 : %s' % str2)


# 원본 리스트, 수정된 리스트 각각 출력
str1 = input('원본 리스트 입력 : ')  # ['abcba','abAb']
vold = input('찾을 문자열 입력 : ')
vnew = input('바꿀 문자열 입력 : ')






# while 문
i = 1

while i < 11 :
    print(i)
    i = i + 1

# [ 연습 문제 ]
# 1부터 100까지의 합
    
i = 1
vsum = 0

while i < 101 : 
    vsum = vsum + i
    i = i + 1



# if문
v1 = 10

if v1 > 3 :
    print('True')
    
if v1 > 3 :
    print('True')
else :
    print('False')
    
v1 = 15    
if v1 > 30 :
    print('30보다 큼')
elif v1 > 10 :
    print('10~30')
else :
    print('10보다 작음')    
        
# eval : 전달된 문자열을 명령어로 해석, 처리를 도와주는 함수
v1='print(1)'
eval(v1)    

v2='(1+2+10)/3'    
eval(v2)    
    
# [ 연습 문제 - 계산기 프로그램 ]
# 종합 계산기 프로그램

ans = int(input('1. 입력한 수식 계산\n2. 두 수 사이의 합계 : '))

if ans == 1:
    str1 = input('계산할 수식을 입력하세요 : ')
    print('%s의 결과는 %s입니다' % (str1, eval(str1)))
    
else :
    no1 = int(input('첫번째 숫자를 입력하세요 : '))
    no2 = int(input('두번째 숫자를 입력하세요 : '))
    
    vsum = 0
    for i in range(no1, no2+1) :
        vsum = vsum + i

    print('%d+...+%d는 %d입니다.' % (no1, no2, vsum))
