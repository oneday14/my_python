# ANN - cancer data
# 1. module loading
run profile1
import tensorflow as tf
import keras

from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder

# 2. data loading
df_cancer = pd.read_csv('cancer.csv')

cancer_x = df_cancer.iloc[:,2:].values
cancer_y = df_cancer.iloc[:,1].values

# 3. data 변환
# 1) Y값 숫자 변환 및 더미 변수 변경
m_label = LabelEncoder()
m_label.fit(cancer_y)
cancer_y_tr = m_label.transform(cancer_y)

cancer_y_tr = np_utils.to_categorical(cancer_y_tr)

# 2) scaling
m_sc = standard()
m_sc.fit(cancer_x)
cancer_x_sc = m_sc.transform(cancer_x)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(cancer_x_sc,
                                                    cancer_y_tr,
                                                    random_state=0)

# 5. ANN 모델 생성
nx = train_x.shape[1]

model = Sequential()
model.add(Dense(15, input_dim = nx, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 25, batch_size = 1)

model.evaluate(test_x, test_y)[1]

# 6. 모델 저장 및 loading
model.save('model_ann_cancer2.h5')

from keras.models import load_model
model = load_model('model_ann_cancer.h5')

# 7. 모델 평가
model.evaluate(test_x, test_y)[1]  # 97.9

# 8. 다른 모델과 비교
m_rf = rf()
m_rf.fit(train_x, train_y)
m_rf.score(test_x, test_y)         # 97.2

# 9. 모델 시각화
print(model.summary())

# plot_model 함수를 사용한 layer 이미지 출력
# 1. window용 graphviz 설치 및 설치 경로 path 등록
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# 1) GRAPHVIZ_DOT 환경 변수 생성 : graphviz설치위치/bin
# C:\Program Files (x86)\graphviz-2.38\release\bin

# 2) PATH에 경로 추가 
# C:\Program Files (x86)\graphviz-2.38

# 2. python용 graphviz 설치
pip install graphviz
conda install graphviz

# 3. python용 pydot 설치
pip install pydot

from keras.utils import plot_model
plot_model(model, to_file='model_ann_cancer.png', 
                  show_shapes=True,
                  show_layer_names=True)


# ANN regressor - boston data 
# 0. metric function define 및 module loading
import tensorflow as tf
import keras

from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder

def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

# 1. data loading
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

# 2. scaling
m_sc = standard()
m_sc.fit(df_boston_et_x)
df_boston_et_x = m_sc.transform(df_boston_et_x)

# 3. data split
train_x, test_x, train_y, test_y = train_test_split(df_boston_et_x,
                                                    df_boston_et_y,
                                                    random_state=0)

# 4. 모델 생성
nx = df_boston_et_x.shape[1]

model = Sequential()
model.add(Dense(52, input_dim = nx, activation = 'relu'))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = ['mean_squared_error', mse,
                         r_square, r_square_loss])

# 5. 자동 학습 중단(EarlyStopping) 적용 
from keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor="mean_squared_error", 
                              patience=10, 
                              verbose=1, 
                              mode='auto')

result = model.fit(train_x, train_y, epochs = 3000, batch_size = 10,
          validation_data=(test_x, test_y), callbacks=[earlystopping])

model.fit(train_x, train_y, epochs = 3000, batch_size = 10,
           validation_split=0.25, callbacks=[earlystopping])


# 6. 모델 시각화
# 1) 모델 요약
model.summary()

# 2) layer 시각화
from keras.utils import plot_model
plot_model(model, to_file='model_ann_boston.png', 
                  show_shapes=True,
                  show_layer_names=True)



# 7. 모델 평가
result.history.keys() # 모델 정보 확인

plt.plot(result.history['loss'], label = 'train')
plt.plot(result.history['val_loss'], label = 'test', c='red')
plt.legend()

plt.plot(result.history['r_square'], label = 'train')
plt.plot(result.history['val_r_square'], label = 'test', c='red')
plt.legend()

model.evaluate(test_x, test_y)     # 전체 평가 지표 출력
model.evaluate(test_x, test_y)[3]  # r_square: 84.53

# loss, metrics에 전달되는 모든 값 출력

# 실제값과 예측값의 비교
v_y_pre = model.predict(test_x).flatten()

for i in range(0,len(test_y)) :
    print('실제값 : %s, 예측값 : %s' % (test_y[i], v_y_pre[i]))

          
# 선형 회귀와 비교
m_reg = LinearRegression()
m_reg.fit(train_x, train_y)
m_reg.score(test_x, test_y)        # 60.75




