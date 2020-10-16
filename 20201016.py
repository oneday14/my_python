# [ 연습 문제 ]
# cancer data를 knn 모델로 예측,
# 의미있는 interaction이 있다면 추가된 이후 예측률 변화 확인
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

train_x, test_x, train_y, test_y = train_test_split(df_cancer.data, 
                                                    df_cancer.target, 
                                                    random_state=0)

# 1. 전체 data 학습
m_knn = knn(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)         # 93

# 2. scaling data 학습
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()

m_sc.fit(train_x)
train_x_sc = m_sc.transform(train_x)
test_x_sc  = m_sc.transform(test_x)

m_knn2 = knn(5)
m_knn2.fit(train_x_sc, train_y)
m_knn2.score(test_x_sc, test_y)      # 95

# 3. 전체 interaction 학습
m_poly = poly(2)
m_poly.fit(train_x_sc)
train_x_sc_poly = m_poly.transform(train_x_sc)
test_x_sc_poly  = m_poly.transform(test_x_sc)
col_poly = m_poly.get_feature_names(df_cancer.feature_names)

m_knn3 = knn(5)
m_knn3.fit(train_x_sc_poly, train_y)
m_knn3.score(test_x_sc_poly, test_y)      # 90

# 4. 선택된 interaction 학습
m_rf = rf(random_state=0)
m_rf.fit(train_x_sc_poly, train_y)
   
s1 = Series(m_rf.feature_importances_ , index = col_poly)
s1.sort_values(ascending=False)

train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).loc[:, col_selected]
test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).loc[:, col_selected]

m_knn4 = knn(5)
m_knn4.fit(train_x_sc_poly_sel, train_y)
m_knn4.score(test_x_sc_poly_sel, test_y)      # 90

## 전진 선택법
l1 = s1.sort_values(ascending=False).index

collist=[]
df_result=DataFrame()

for i in l1 : 
    collist.append(i)
    train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).loc[:, collist]
    test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).loc[:, collist]

    m_knn5 = knn(5)
    m_knn5.fit(train_x_sc_poly_sel, train_y)
    vscore = m_knn5.score(test_x_sc_poly_sel, test_y)  
    
    df1 = DataFrame([Series(collist).str.cat(sep='+'), vscore], index=['column_list', 'score']).T
    df_result = pd.concat([df_result, df1], ignore_index=True)
    
df_result.sort_values(by='score', ascending=False)
