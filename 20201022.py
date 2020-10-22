# 이미지 인식/ 분석 : PCA + knn
# sklearn에서의 이미지 데이터 셋
- 2000년 초반 이후 유명인사 얼굴 데이터
- 전처리 속도를 위해 흑백으로 제공
- 총 62명의 사람(target)의 얼굴을 여러장 촬영한 데이터 제공
- 총 3023개(행의 수)의 이미지 데이터, 87X65(컬럼수) 픽셀로 규격화 제공

# 1. 데이터 로딩 및 설명
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

people.keys()        # ['data', 'images', 'target', 'target_names', 'DESCR']
people.data.shape    # 2차원 (3023, 5655)
people.images.shape  # 3차원 (3023, 87, 65)

people.images[0,:,:]

people.data[0,:]                        # 첫번째 이미지의 변환된 값
people.target[0]                        # 학습용 Y값(숫자로 변환된)
people.target_names[people.target[0]]   # 원래 Y값 : 'Winona Ryder'

people.data[0,:].min()
people.data[0,:].max()


# RGB 변환된 sample data를 다시 이미지로 변환
run profile1
fig, ax = plt.subplots(1,3, figsize=(15,8))

ax[0].imshow(people.images[0,:,:])
ax[1].imshow(people.images[100,:,:])
ax[2].imshow(people.images[1000,:,:])

name1 = people.target_names[people.target[0]]
name2 = people.target_names[people.target[100]]
name3 = people.target_names[people.target[1000]]

ax[0].set_title(name1)
ax[1].set_title(name2)
ax[2].set_title(name3)
 


# 2. 각 클래스별 균등 추출
people.target

np.bincount(people.target) 
np.bincount([1,1,1,2,2])   # [0, 3, 2]
                           #  0의 개수, 1의 개수, 2의 개수

# Y값이 10인 대상을 50개만 추출
np.where(people.target == 10)         # 조건에 맞는 행 번호 리턴

np.where(people.target == 10)[0]      # tuple에서 array 추출 위해
np.where(people.target == 10)[0][:50] # 처음부터 50개 추출

# 전체 Y에 대한 최대 50개 추출
v_nrow = []
for i in np.unique(people.target):
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)

len(v_nrow)  # 각 클래스별 최대 50개 추출 후 data set 크기 : 2063

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# 참고 : array 확장-----------------------
a1 = np.array([1,2,3])
a2 = np.array([4,5,6])
a1.append?                          # 불가

pd.concat([Series(a1),Series(a2)])
np.hstack([a1,a2])
list(a1) + list(a2)
# ----------------------------------------

# 3. train, test 분리
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state=0)

# 4. knn 모델 적용
m_knn = knn(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)   # 19.38

v_pre = m_knn.predict(test_x[0].reshape(1, -1))   # 2차원 형식으로 전달
people.target_names[v_pre][0]   # 예측 값
people.target_names[test_y[0]]  # 실제 값

# 5. 예측값과 실제값의 시각화
fig, ax = plt.subplots(1,2, figsize=(15,8))
ax[0].imshow(test_x[0].reshape(87,65))
ax[1].imshow(people.images[people.target == 23][0])


plt.rc('font',family='Malgun Gothic')

ax[0].set_title('예측값 : Colin Powell')
ax[1].set_title('실제값 : Jack Straw')


# 6. 기타 튜닝
# 스케일링
m_sc = standard()
m_sc.fit(train_x)
train_x_sc = m_sc.transform(train_x)
test_x_sc = m_sc.transform(test_x)

v_score_tr = [] ; v_score_te = []

for i in range(1,11) : 
    m_knn = knn(i)
    m_knn.fit(train_x_sc, train_y)
    v_score_tr.append(m_knn.score(train_x_sc, train_y))
    v_score_te.append(m_knn.score(test_x_sc, test_y))
    
plt.plot(v_score_tr, label='train_score')
plt.plot(v_score_te, c='red', label='test_score')

plt.legend()


# 7. PCA로 변수가공
m_pca = PCA(n_components=100, whiten=True)
m_pca.fit(train_x_sc)
train_x_sc_pca = m_pca.transform(train_x_sc)
test_x_sc_pca = m_pca.transform(test_x_sc)

train_x_sc_pca.shape   # (1547, 100)
test_x_sc_pca.shape    # (516, 100)

m_knn = knn(3)
m_knn.fit(train_x_sc_pca, train_y)
m_knn.score(test_x_sc_pca, test_y)


# 참고 : 이미지에서 RGB 추출
import imageio
im = imageio.imread('cat2.gif')

im.shape   # (200, 200, 4)

