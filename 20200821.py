# 1. 구구단 출력(중첩 for문)
# 2X 1 = 2 3X 1 = 3 ... 9X 1 = 9
for n in range(2,10) :
    print('# %d단 #' % n, end='   ')
print()

for j in range(1,10) :
    for i in range(2,10) :
        print('%dX %d = %2d' % (i,j,i*j), end=' ')
    print()    

# 2X 1 = 2 3X 1 = 3 ... 9X 1 = 9
# 2X 2 = 4 3X 2 = 6 ... 9X 2 = 18    


# 2. 별 출력(while문)
# 1) 수학적 수열식 사용    
v1 = '  '
v2 = '\u2605'

i=1
while i < 10 :
    if i < 6 :
        print(v1*(5-i) + v2*(2*i - 1))
    else :
        print(v1*(i-5) + v2*(2*(10-i) - 1))
    i = i + 1

#                        i     공백      별
# print(v1*4 + v2*1)     1     5-i     2i-1
# print(v1*3 + v2*3)     2
# print(v1*2 + v2*5)     3
# ...
# print(v1*1 + v2*7)     6     i-5     2(10-i)-1
# print(v1*2 + v2*5)     7
# print(v1*3 + v2*3)     8
# print(v1*4 + v2*1)     9   

# 2) 변수 2개 사용
i = 1
j = 7
while (i < 10 or j > 0):
    if i < 10 :
        print('%9s' % ('\u2605' * i))
        i += 2
    else :
        print('%9s' % ('\u2605' * j))
        j -= 2
                
    
# 3. 사용자로부터 하나의 단어를 입력받고 회문여부 판별
# 회문이란 앞뒤로 똑같이 생긴 단어를 의미
# 예) allalla
v3 = 'abccba'     
v3 = 'allalla'    

v3[0] == v3[-1]    
v3[1] == v3[-2]     
v3[2] == v3[-3]     

  

# 프로그램 작성
vstr = input('회문을 판별할 문자열을 입력하세요 : ')  
vcnt = len(vstr) // 2         # 반복횟수

vre=0    
for i in range(0,vcnt) :
    if vstr[i] == vstr[-(i+1)] :  # llacocbll
        vre = vre + 0         # 앞,뒤 비교가 같으면 0을 더하고
    else :
        vre = vre + 1         # 다르면 1(틀린횟수)을 누적 합
        
if vre == 0 :
    print('회문입니다')
else :
    print('회문이 아닙니다')
    
########## 여기까지는 복습입니다. ##########

# 반복 제어문
# 1. continue : R의 next와 비슷, 특정 반복문만 skip
# 2. break : R도 break, 반복문 자체 종료
# 3. exit : R의 quit과 비슷, 프로그램 자체 종료
# 4. pass : 빈 줄로 명령어 전달 불가 시 대체

# ex) continue
for i in range(1,11) :
    if i == 5 :
        continue        # 5일때 continue를 만나 다음의 구문 skip
    print(i)
print('프로그램 끝')  # 정상 출력

# ex) break
for i in range(1,11) :
    if i == 5 :
        break        # 5일때 break를 만나 즉시 반복문 종료
    print(i)    
    
print('프로그램 끝')  # 정상 출력    

# ex) exit
for i in range(1,11) :
    if i == 5 :
        exit(0)      # 5일때 exit를 만나 즉시 프로그램 종료
    print(i)   
    
print('프로그램 끝')  # 정상 출력 X

# ex) pass
v1 = 'q1'

if v1 == 'q' :
    pass
else :
    print('잘못된 입력')

if v1 != 'q' :
    print('잘못된 입력')

# for  : 
#     for :
#         if :
#             break
#         print        # break로 인해 수행 X
#     print(values)    # break와 상관없이 계속 반복 수행
# print                # 외부 for문이 끝나면 정상 수행 


# 1부터 100까지 누적합을 더하다가 누적합이 3000이 넘는 지점을 출력
# (해당 지점과 해당지점까지의 누적합 출력)
# i
# 30   2800
# 31   3200

vsum=0
for i in range(1,101) :
    vsum = vsum + i
    if vsum >= 3000 :
        break
    
print('3000이 넘는 지점 : %d' % i)
print('1+...+ %d = %d' % (i,vsum))


# 연습 문제
# 사용자로부터 값을 입력받아 불규칙한 중첩 리스트를 만드려고 한다
# 단, 사용자가 종료코드(q)를 입력하면 즉시 종료 후
# 입력된 불규칙한 리스트를 출력

# l1 = [[1,2,3,4], [5,6], [7,8,9]]

# 1 2 3 4
# 5 6
# 7 8 9

i=1
outlist=[]
while 1 :
    vstr = input('%d번째 원소를 입력하세요 : ' % i)
    if vstr=='q' : 
        break
    inlist = vstr.split(',')
    outlist.append(inlist)
    i=i+1

for j in outlist :
    for z in j :
        print(z, end=' ')
    print()    



# 사용자 정의 함수
# 1. lambda : 복잡한 프로그래밍 처리 없는 간단한 리턴 가능한 축약형 문법
f1 = lambda x : x + 1

# 2. def : 복잡한 프로그래밍 처리 가능한 문법
def f2(x) :
    return x + 1

f2(10)    

def f2(x=0,y=0) :
    return x + y

f2(10,1)  

# 사용자 정의 함수 유용 표현식
# 1. for문에 여러 인자 동시 전달 : zip
l1 = [1,2,3]
l2 = [10,20,30]

for i,j in zip(l1,l2) :
    print(i+j)

# 2. 전역 변수 선언 : global
# 1)
v1 = 1    

def f1() :
    return v1    
    
f1()    


# 2) 전역변수보다 지역변수 우선순위(함수 실행시 선언하므로)
v1 = 1    

def f1() :
    v1=10
    return v1    
    
f1() 

# 3) 지역변수의 전달 불가
def f3() :
    v10 = 10
    return v10

def f4() :
    return v10

f3()  # 10  
f4()  # name 'v10' is not defined
v10   # name 'v10' is not defined

# 4) 지역변수의 전역변수 설정
def f3() :
    global v10     # v10 <<- 10 (in R)
    v10 = 10       # export v10 = 10 ( v10=10 ; export v10)
    return v10

def f4() :
    return v10

f3()  # 10  
f4()  # 10
v10   # 10



# [ 연습 문제 ]
# 다음의 리스트에서 ';'로 분리된 첫번째 값을 추출하는 사용자 정의 함수 생성 및 적용
l1=['a;b;c','A;B;C']

def f3(x) :
    return x.split(';')[0]

f3('a;b;c')

list(map(f3, l1))

# 파일 입출력
# 1. open : 파일을 열어 파일의 내용을 파이썬 메모리영역(커서)에 저장
# 2. fetch : 커서에 저장된 데이터를 인출(형상화)
# 3. close : 커서의 영역 해제, close하지 않을 경우 메모리누수 현상 발생 가능성

# 1. read
# 1)
c1 = open('read_test1.txt')

v1 = c1.readline()
print(v1)

v2 = c1.readline()
print(v2)

c1.close()


# 2)
c1 = open('read_test1.txt')

while 1 :
    v1 = c1.readline()
    if v1 == '' :
        break
    print(v1)

c1.close()

# 3)
c1 = open('read_test1.txt')

while 1 :
    v1 = c1.readline()
    if v1 == '' :
        break
    print(v1, end='')

c1.close()

# 4)
c1 = open('read_test1.txt')

outlist = c1.readlines()

c1.close()

outlist

# [ 연습 문제 ]
# 다음의 사용자 정의 함수 생성
# 외부 파일을 불러와 중첩 리스트로 저장하는 함수

# f_read_txt(file, sep=';', fmt='int')

# sol1) fmt 인자에 함수 형식으로 전달
def f_read_txt(file, sep=';', fmt=int) :
    c1 = open(file)
    l1 = c1.readlines()
    c1.close()
    
    outlist=[]
    
    for i in l1 :
        l2 = i.strip().split(sep)
        inlist=[]
        for j in l2 :
            inlist.append(fmt(j))     
        outlist.append(inlist)   
    
    return outlist

f_read_txt('read_test1.txt', sep=' ', fmt=float) 

# sol2) fmt 인자에 문자열 형식으로 전달
def f_read_txt(file, sep=';', fmt='int') :
    c1 = open(file)
    l1 = c1.readlines()
    c1.close()
    
    outlist=[]
    
    for i in l1 :
        l2 = i.strip().split(sep)
        inlist=[]
        for j in l2 :
            vstr = fmt + '(' + str(j) + ')' 
            inlist.append(eval(vstr))     
        outlist.append(inlist)   
    
    return outlist

f_read_txt('read_test1.txt', sep=' ', fmt='int') 


fmt='int'
fmt + '(' + '1' + ')'   #'int(1)'
eval(fmt + '(' + '1' + ')')

# 2. write
l1 = [[1,2,3],[4,5,6]]

# 1)
c1=open('write_test1.txt','w')
c1.writelines(str(l1))
c1.close()                      # [[1, 2, 3], [4, 5, 6]]

# 2) 
c1=open('write_test1.txt','w')
for i in l1 :
    c1.writelines(str(i) + '\n')
c1.close() 
