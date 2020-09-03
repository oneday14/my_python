# 1. 1부터 증가하는 3 X 4 X 5 배열 생성 후
run profile1       # profile1.py 생성 후 실행 가능

arr1 = np.arange(1,61).reshape(3,4,5)

# 1) 모든 값에 짝수는 *2를 홀수는 *3을 연산하여 출력
# sol1) np.where
np.where(arr1%2==0, arr1*2, arr1*3)   # 벡터 연산 가능

# sol2) if문을 사용한 치환
# --
if arr1%2 == 0 :    # 불가
    arr1*2
else : 
    arr1*3

# -- for + return : 여러번 반복될때마다 return이 수행되는 구조
def f1(x) :
    for i in x :
        if i%2 == 0 :
            return i*2
        else :
            return i*3
        
f1(3)           # 에러 발생
f1([3,6,7])     # 하나만 리턴(return은 한 번만 수행되므로)

# =============================================================================
# for i in 3 :    # for문에 숫자 스칼라 전달 불가
#     print(i)
# 
# for i in [3] :
#     print(i)
# 
# for i in range(0,3) :
#     print(i)
# =============================================================================

# -- 
def f1(x) :
    outlist=[]
    for i in x :
        if i%2 == 0 :
            outlist.append(i*2)
        else :
            outlist.append(i*3)
    return outlist        

f1([1,2,3])         # 출력 가능
f1(arr1)            # 에러발생, arr1[0]이 for문에 i에 전달되므로

# =============================================================================
# for i in arr1 :     # 각 층마다 출력, 3번 반복
#     print(i)
# 
# for i in [[1,2,3],[4,5,6]] :   # 6번 반복 X, 2번 반복
#     print(i)
# =============================================================================

# =============================================================================
# [ 정리 - for문에서 객체가 반복되는 형태 ]
# for i in '12345' :               # for문에 문자열 스칼라가 들어가면
#     print(i)                     # 문자열을 각 글자로 분리 후 반복 수행
# 
# for i in obj :                   # i = obj[0], i = obj[1]
#     command
# =============================================================================

# 2) 각 층의 첫번째 세번째 행의 두번째 네번째 컬럼 선택하여 NA로 치환
arr1[:,[0,2],[1,3]]                     # point indexing 
arr1[:,[0,2], :][:, :, [1,3]]           # 가능
arr1[np.ix_([0,1,2],[0,2],[1,3])]       # 가능

arr1[np.ix_([0,1,2],[0,2],[1,3])] = NA  # 에러

arr1 = arr1.astype('float')
arr1[:,[0,2], :][:, :, [1,3]]     = NA  # 중첩 색인시 원본 수정 불가

arr1[np.ix_([0,1,2],[0,2],[1,3])] = NA  # 정상 수행

# =============================================================================
# [ 참고 - 객체 수정 여부 ]
#
# l1 = [[1,2,3],[4,5,6]]
# [ x[0] for x in l1 ] = [10,20]  # 원본으로부터 파생된 객체 수정 불가
# 
# a1 = np.array(l1)
# a1[:,0]= [10,20]                # 원본의 일부 수정 가능
#
# =============================================================================



# =============================================================================
# [ 참고 - astype 메서드를 사용한 형 변환 유형 ]
#
# 1. int/float/str 의 형 유형 전달 방식
# arr1.astype('float')
# 
# 2. type code 전달 방식/size 지정 가능
# arr1.astype('S40')    # string 40 bytes
# arr1.astype('U40')    # unicode 40 bytes
# arr1.astype('f')      # float
# arr1.astype('i')      # int
# =============================================================================


# =============================================================================
# [ 참고 : np.ix_ 함수의 인자 전달 방식 ]
# - 리스트만 전달 가능
# - 정수 스칼라 전달 불가, 리스트 안에 삽입 후 전달
# - 슬라이스 색인 형태 전달 불가, 리트로 전달
# =============================================================================
arr1[np.ix_(  0,[0,2],[1,3])]     # 불가
arr1[np.ix_([0],[0,2],[1,3])]     # 가능

arr1[np.ix_(0:2  ,[0,2],[1,3])]   # 불가
arr1[np.ix_([0,1],[0,2],[1,3])]   # 가능

# 3) 위의 수정된 배열에서 NA의 개수 확인
np.isnan(arr1).sum()

# 4) 층별 누적합 확인
arr1.cumsum(0)

# 2. emp.csv 파일을 array 형식으로 불러온 뒤 다음 수행(컬럼명은 제외)
arr2 = np.loadtxt('emp.csv', skiprows=1, dtype='str', delimiter=',')

# 1) 이름이 smith와 allen의 이름, 부서번호, 연봉 출력
# -- or
(arr2[:,1] == 'smith') or (arr2[:,1] == 'ALLEN')  # 벡터 연산 불가

('smith' == 'smith') or ('smith' == 'ALLEN')      # 벡터 연산 불가

f1 = lambda x : (x == 'smith') or (x == 'ALLEN') 
list(map(f1, arr2[:,1]))

# -- |
(arr2[:,1] == 'smith') | (arr2[:,1] == 'ALLEN')   # 벡터 연산 가능

# in
'smith' in ['smith','ALLEN']    # 스칼라 가능
arr2[:,1] in ['smith','ALLEN']  # 벡터 연산 불가

f2 = lambda x : x in ['smith','ALLEN']
list(map(f2, arr2[:,1]))

# np.in1d
np.in1d(['smith','king','allen'],['smith','allen'])
vbool = np.in1d(arr2[:,1],['smith','ALLEN'])

arr2[vbool, :][:,[1,-1,-3]]
arr2[np.ix_(vbool,[1,-1,-3])]

# =============================================================================
# [ 참고 - 각 언어의 in 연산자 비교 ]
#
# ename in ('smith','allen')     # in sql
# ename %in% c('smith','allen')  # in R
#
# =============================================================================

# 2) deptno가 30번 직원의 comm의 총 합
# sol1) replace + map(X)
v1 = arr2[arr2[:,-1] == '30', -2]

''.replace('','0')
'300'.replace('','0')    # 300으로 리턴 X
'300'.replace('3','0')

v1.replace('','0')

# =============================================================================
# [ 참고 - replace 메서드의 값 치환과 패턴치환 ]
#
# 'abc'.replace('a','A')    # 패턴 치환
# 'abc'.replace('abc','X')  # 패턴 치환
# 'abcd'.replace('abc','X') # 패턴 치환, 값 치환 불가
# 
# if 'abc' == 'abc' :       # 값 치환
#     print('X')
# 
# =============================================================================

# sol2) np.where(O)
arr2[:,-2] = np.where(arr2[:,-2] == '', '0', arr2[:,-2])
arr2[arr2[:,-1] == '30', -2].astype('int').sum()

# 3. professor.csv 파일을 array 형식으로 불러온 뒤 다음 수행(컬럼명은 제외)
arr3 = np.loadtxt('professor.csv', skiprows=1, dtype='str', delimiter=',')

# 1) email_id 출력
arr3[:,-2]
vid = list(map(lambda x : x.split('@')[0], arr3[:,-2]))

# 2) 홈페이지가 없는 사람들은 다음과 같이 출력
'http://www.kic/com/email_id'

# -- np.where 불가 : 'http://www.kic/com/' + vid 을 얻을수 없어서
np.where(arr3[:,-1]=='', 'http://www.kic/com/' + vid , arr3[:,-1])

# -- map
f4 = lambda x,y : 'http://www.kic/com/' + y if x == '' else x
vhpage = list(map(f4,arr3[:,-1],vid))

# 원본 배열의 홈페이지 주소 변경
arr3[:,-1] = arr3[:,-1].astype('U40')
arr3[:,-1] = vhpage                  # 문자열 잘림 현상
arr3[:,-1].dtype                     # U40으로 변경 X, 원본 배열이 U21 이므로

arr3.dtype
arr3 = arr3.astype('U40')
arr3[:,-1] = vhpage

########## 여기까지는 복습입니다. ##########

# profile 설정
# - 매 세션마다 호출해야 하는 여러 모듈을 한번에 호출하기 위한 파일
# - .py 파일로 저장

# 1. 새로운 파일 열기
# 2. 호출할 모듈 나열
# 3. 저장
# 4. 새 세션에서 run profile1으로 호출


# =============================================================================
# 정리
# =============================================================================
# 1. 모듈 호출 방식

# 2. print 출력 형식 / 포맷 변경
v1='a'
print('입력된 값은 v1입니다')
print('입력된 값은 %s입니다' % v1)

'입력된 값은 %s입니다' % v1
'입력된 값은 %.2f입니다' % int('123')

# 3. 형변환 함수 / 메서드

# 4. 파이썬 자료 구조
# 1) 리스트
# 2) 딕셔너리
# 3) 배열
# 4) 데이터 프레임

# 5. 문자열 메서드
# 1) upper, lower, title
# 2) startswith, endswith
# 3) strip, lstrip, rstrip
# 4) find
# 5) replace
# 6) split
# 7) count
# 8) isalpha, isnumeric, isalnum

# 6. lambda + map

# 7. list comprehension

# 8. if문, for문(중첩 for문), while문

# 9. 반복문 제어(continue, break, exit, pass)

# 10. 사용자 정의 함수
# 1) def 함수 생성
# 2) 가변형 인자전달
# 3) 딕셔너리형 인자 전달
# 4) global
# 5) zip

# 11. 리스트 메서드
# 1) append
# 2) extend
# 3) insert
# 4) remove
# 5) pop
# 6) index
# 7) count
# 8) sort
# 9) reverse

# 12. 파일 입출력
# 1) 리스트 입출력
# 2) numpy 입출력

# 13. deep copy
# 1) list deep copy
# 2) array deep copy

# 14. 배열 메서드(dtype, ndim, reshape, shape)

# 15. 배열의 산술연산 및 broadcast

# 16. 배열의 색인

# 17. 배열의 축 번호와 전치

# 18. np.where

# 19. 배열의 집합연산자

# 20. in 연산자
# 1) 문자열 in 연산자 : 문자열 패턴 확인 가능, 벡터 연산 불가
'x' in 'xvg'

# 2) 리스트 in 연산자 : 원소 포함 여부, 벡터 연산 불가
1 in [1,2]

# 3) np.in1d : array 원소 포함 여부, 벡터 연산 가능
np.in1d([1,2,10,11],[1,2])

# 4) isin 메서드(in pandas)
s1 = Series([1,2,3,4,5])
s1.isin([1,2])
ename.isin(['smith','allen'])

# 21. for문의 인자 전달 방식****


# [ 연습 문제 ]
# 1. card_history.txt 파일을 array로 읽고(첫번째 행 제외)
arr1 = np.loadtxt('card_history.txt', delimiter='\t', skiprows=1,
                   dtype='str', encoding='utf8')

# 천단위 구분기호 제거 후 2차원 객체 적용
# 1) 함수의 input이 스칼라인 경우
f1 = lambda x : int(x.replace(',',''))

list(map(f1,arr1))  # 2차원 데이터 셋 적용 불가

arr1[:,0] = list(map(f1, list(arr1[:,0])))
arr1[:,1] = list(map(f1, list(arr1[:,1])))
arr1[:,2] = list(map(f1, list(arr1[:,2])))
....

# 2) 함수의 input이 2차원 array
def f2(x) :
    outlist=[]
    for i in x :
        inlist=[]
        for j in i :
            inlist.append(int(j.replace(',','')))
        outlist.append(inlist)
    return np.array(outlist)                  

arr2 = f2(arr1)

# 3) 함수의 input이 스칼라인 경우 + applymap
DataFrame(arr1).applymap(f1)


# 1) 각 품목별 총 합 출력
arr2[:,1:].sum(axis=0)
DataFrame(arr2[:,1:]).apply(sum, axis=0)

# 2) 의복 지출이 가장 많은 날의 총 지출 금액 출력
arr2[np.argmax(arr2[:,2]), :][1:].sum()


# python에서의 적용함수/메서드
# 1. map 함수
# - 1차원에 원소별 적용 가능
# - 반드시 리스트로만 출력 가능
# - 적용함수의 input은 스칼라 형태
# - 함수의 추가 인자 전달 가능****

# 2. map 메서드(pandas 제공 메서드)
# - 1차원(시리즈)에 원소별 적용 가능
# - 반드시 시리즈만 출력 가능
# - 적용함수의 input은 스칼라 형태
# - 함수의 추가 인자 전달 불가

# =============================================================================
# s1 = Series(['abc','bcd'])
# 
# map(func, *iterables)
# s1.map(arg, na_action=None)   # *iterables
# =============================================================================

# 3. apply
# - 2차원 데이터프레임의 행별, 컬럼별 적용 가능(pandas 제공 메서드)
# - 적용함수의 input은 그룹(여러개 값을 갖는) 형태
# - 함수의 추가 인자 전달 불가(함수의 옵션은 전달 가능)
# - df1.apply(func, **kwds)

# 4. applymap
# - 2차원 데이터프레임의 원소별 적용 가능(pandas 제공 메서드)
# - 출력 결과 데이터프레임
# - 적용함수의 input은 스칼라 형태
# - df1.applymap(func), 함수의 추가 인자 전달 불가

pd.read_csv(filename, sep=',', engine='python')

df1 = pd.read_csv('emp.csv')
df1.dtypes     # 컬럼별 데이터 타입 확인

df1.EMPNO      # 컬럼 선택
