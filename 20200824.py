# 1. 아래와 같이 변수로 선언된 중첩 리스트를 외부 파일로 저장하는 함수 생성
# f_write_txt(l1, 'write_test1.txt', sep=' ', fmt='%.2f')
l1=[[1,2,3,4],[5,6,7,8]]

'%d' % 1

def f_write_txt(obj, file, sep=' ', fmt='%s') :
    
    c1 = open(file,'w')
    
    for i in obj :
        vstr = ''
        for j in i : 
            vfmt = fmt % j
            vstr = vstr + vfmt + sep
        vstr = vstr.rstrip(sep)
        c1.writelines(vstr + '\n')
        
    c1.close()

f_write_txt(l1, 'write_test3.txt', sep=';', fmt='%.2f')

# 2. oracle instr과 같은 함수 생성(없으면 -1 생성)
# f_instr(data,pattern,start=0,  # 0부터 스캔
#                          n=1)      # 첫번째 발견된 pattern의 위치 확인

v1='1#2#3#4#'
v1.find('#')     # 처음(0)부터 스캔해서 첫번째 발견된 # 위치
v1.find('#',2)   # 2 위치부터 스캔해서 첫번째 발견된 # 위치

# 1) n번째 발견된? = 적어도 찾고자 하는 문자열이 n개 포함되어 있다는 의미
v1.count('#') >= n 조건 성립

# 2) n번째 발견된 위치 리턴?instr(v1, '#', 2, 3)
# step1) 첫번째 '#'을 시작위치에서 찾고
v1.find('#',2)  # 3

# step2) 위 위치 다음부터 다시 스캔하여 '#'의 위치 확인
v1.find('#',3+1)  #

def f_instr(data,pattern,start=0,n=1)  :
    vcnt = data.count(pattern)
    if vcnt < n :
        position = -1
    else :
        for i in range(0,n) :
            position = data.find(pattern,start)
            start = position+1
    return position

f_instr('1#2#3#4#','#',start=0,n=1)   # 1
f_instr('1#2#3#4#','#',start=0,n=2)   # 3
f_instr('1#2#3#4#','#',start=0,n=10)  # -1
f_instr('1#2#3#4#','#',start=2,n=1)   # 3
f_instr('1#2#3#4#','#',start=2,n=3)   # 7

########## 여기까지는 복습입니다. ##########

# =============================================================================
# 자료구조
# 1. 리스트**
# 2. 튜플
# 3. 딕셔너리**
# 4. 세트
# 5. 배열**
# 6. 데이터프레임**
# =============================================================================

# 튜플
# - 리스트와 같은 형태이나 수정 불가능한 읽기 전용 객체
# - 원소 삽입, 수정, 삭제 불가
# - 객체 자체 삭제 가능

l1 = [1,2,3,4]
t1 = tuple(l1)
t2 = (1,2,3,4)

type(t1)
type(t2)

t1.append(5)         # 오류
t1[0] = 10           # 오류
del(t1[0])

del(t1)


# 딕셔너리
# - key-value 자료구조
# - R에서의 리스트와 비슷
# - 2차원 X
# - pandas의 Series, Dataframe의 기본 자료 구조

# 1. 생성
d1 = {'col1':[1,2,3,4],
      'col2':['a','b']}

type(d1)

# 2. 색인
d1['col1']
d1.get('col1')

# 3. 수정
l1[4] = 5            # list의 out of range 추가 불가
d1['col3'] = [1,2]   # key 추가
d1['col1'][0] = 10

from numpy import nan as NA

d1['col1'] = NULL    # R방식, 파이썬에서는 불가
d1['col1'] = NA      # NA로 대체, 키 삭제 불가

del(d1['col1'])      # 키 삭제 가능

# 4. 메서드
d1.keys()
d1.values()
d1.items()


# 연습문제) 다음의 리스트와 딕셔너리를 참고하여 전화번호를 완성 : 02)345-4958
l1=['345-4958','334-0948','394-9050','473-3853']
l2=['서울','경기','부산','제주']

area_no={'서울':"02",'경기':"031",'부산':"051" ,'제주':"064"}

l3=[]
for i,j in zip(l1,l2) :
    l3.append(area_no.get(j) + ')' + i)


f1 = lambda x, y : area_no.get(y) + ')' + x
list(map(f1,l1,l2))

# 세트
# - 딕셔너리의 키 값만 존재
# - 키 값은 중복될 수 없으므로 중복 불가능한 객체 생성시 사용

s1 = set(d1)
type(s1)

s2 = {'a','b','c'}
type(s2)

s3 = {'a','b','c','c'}


# [ 연습 문제 : 로또 번호 생성 프로그램 ]
import random
random.randrange(1,46)

# 1)
lotto=[]
while len(lotto) < 6 :
    vno = random.randrange(1,46)
    if vno in lotto :
        pass
    else : 
        lotto.append(vno)

lotto.sort()    

vstr = ''
for i in lotto :
    vstr = vstr + str(i) + ' '
    
print ("추첨된 로또 번호 ===> %s" % vstr)



lotto=[]
while len(lotto) < 6 :
    vno = random.randrange(1,46)
    lotto = list(lotto)
    lotto.append(vno)
    lotto = set(lotto)

lotto = list(lotto)
lotto.sort()

vstr = ''
for i in lotto :
    vstr = vstr + str(i) + ' '
    
print ("추첨된 로또 번호 ===> %s" % vstr)



# 리스트 내포 표현식(list comprehension)
# - 리스트의 반복 표현식(for)의 줄임 형식
# - lambda + mapping과 비슷
l1 = [1,2,3,4]

# 1. [리턴대상 for 반복변수 in 반복대상 ]
[i*3 for i in l1]

list(map(lambda i : i*3, l1))

l2=[]
for i in l1 :
    l2.append(i*3)


# 2. [참리턴대상 for 반복변수 in 반복대상 if 조건 ]
l2=[1,2,3,4,5]

[i*10 for i in l2 if i > 3]          # 조건에 만족하지 않는 대상은 생략
[i*10 for i in l2 if i > 3 else i*5] # else 리턴 불가

list(map(lambda i : i*10 if i > 3 else None, l2))

# 3. [참리턴대상 if 조건 else 거짓리턴대상 for 반복변수 in 반복대상 ]
[i*10 if i > 3 else i*5 for i in l2]
list(map(lambda i : i*10 if i > 3 else i*5, l2))




# [ 연습 문제 ]    
sal = ['9,900','25,000','13,000']
addr = ['a;b;c','aa;bb;cc','aaa;bbb;ccc']
comm = [1000,1600,2000]

# 1) sal의 10% 인상값 출력
list(map(lambda x : round(int(x.replace(',','')) * 1.1,2), sal))

[ round(int(x.replace(',','')) * 1.1,2) for x in sal ]

new_sal=[]
for x in sal :
    new_sal.append(round(int(x.replace(',','')) * 1.1,2))

# 2) addr에서 각 두번째 값(b,bb,bbb) 출력
list(map(lambda x : x.split(';')[1] , addr))

[ x.split(';')[1] for x in addr ]

# 3) comm이 1500보다 큰 경우 'A', 아니면 'B' 출력
list(map(lambda x : 'A' if x > 1500 else 'B', comm))

[ 'A' if x > 1500 else 'B' for x in comm ]


# deep copy(깊은 복사)
# - 객체의 복사가 원본과 전혀 다른 공간(메모리)을 차지하는 형태로 복사되는 경우
# - 파이썬의 기본 복사는 얕은 복사인 경우 많음(같은 메모리 영역 참조)

l1=[1,2,3,4]
l2=l1         # 얕은 복사
l3=l1[:]      # 깊은 복사

l2[0] = 10    # l2 수정
l1[0]         # 10으로 변경되어 있음

l3[1] = 20    # l3 수정
l1[1]         # 20으로 변경되지 X


id(l1)        # 1956891696584
id(l2)        # 1956891696584
id(l3)        # 1956891347784


# 패킹과 언패킹
a1 = 1,2,3       # 튜플 객체로 생성
v1,v2,v3 = a1    # v1=a1[0] ; v2=a1[1] ; v3=a1[2]


# [ 연습 문제 : 딕셔너리 음식 궁합 ]
food_dic = {'치킨':'치킨무',
            '라면':'김치',
            '떡볶이':'순대',
            '짜장면':'짬뽕',
            '피자':'콜라',
            '맥주':'소주',
            '삼겹살':'소고기'}

while 1 :
    flist = list(food_dic.keys())
    ans = input('%s중 좋아하는 음식은? : ' % flist)
    if ans == '끝' :
        break
    elif ans in flist :
        print('<%s> 궁합 음식은 <%s>입니다.' % (ans, food_dic.get(ans)))
    else :
        print('그런 음식은 없습니다. 확인해보세요')


# =============================================================================
# 사용자 정의 함수
# 1. def를 사용한 사용자 정의 함수 생성
# 2. global : 전역변수 설정
# 3. zip : 동시 여러 객체 전달
# 4. 가변형 인자 전달 방식
# 5. 딕셔너리형 인자 전달 방식
# 6. 모듈        
# =============================================================================

# 모듈
# - 함수의 묶음
# - .py 프로그램으로 저장
# - import 호출 가능

# 가변형 인자 전달 방식
# - 함수의 인자의 전달 횟수에 제한 두지 X
# - for문 필요

def f1(x, *iterable) :
    for i in iterable :
        print(i)
        
f1(1,2,3,4,5,6)

# [ 연습 문제 ]
# 나열된 값의 누적곱을 리턴하는 함수 생성

def f_prod(*iterable) :
    vprod = 1
    for i in iterable :
        vprod = vprod * i
    return vprod    
        
f_prod(1,6,9)

# 딕셔너리형 인자 전달 방식
f_add(1,2,fmt='%.2f')

def f3(x, **dict) : 
    for i in dict.keys() :
        print(dict.get(i))

f3(1, v1=1, v2=2, v3=3)


# [ 연습 문제 ]
# 두 수를 전달받아 두 수의 곱을 구하여 리스트에 저장,
# 저장된 값은 숫자가 큰 순서대로 정렬하여 출력하도록 하는 사용자 정의함수 생성.
# 단, 사용자 정의함수에 두 수 이외의 reverse라는 키워드 인자를 입력 받도록 하자.
fprod(L1,L2,reverse=True)

def fprof(x,y,**dict) :
    vresult=[]
    for i,j in zip(x,y) :
        vresult.append(i*j)
    vresult.sort(reverse=dict['reverse'])    
    return vresult
    

fprof([4,7,9,2],[8,9,3,6], reverse=True)


# 패키지
# - 함수의 묶음이 모듈
# - 모듈의 집합이 패키지
# - 패키지는 하나의 폴더

# from 패키지명.모듈명 import 함수명

from p1.my_func import *

