# machine learning > deep learning

# deep learning : 신경망(NN) 구조를 본따 만든 모델
# ANN, CNN, RNN, ....

# - node : 뉴런
# - layer : 여러 뉴런이 모여있는 단위
#  1) input layer : 외부 자극을 받아들이는 뉴런의 집합
#  2) hidden layer : 중간 사고를 담당하는 뉴런의 집합 ***
#  3) output layer : 최종 판단을 출력하는 뉴런의 집합

# deep learning 구현 
# - tensorflow
# - keras

# 설치
# C:\Users\KITCOOP> pip install tensorflow
# C:\Users\KITCOOP> pip install keras


# ANN 모델을 통한 iris data의 예측(꽃의 품종)
# 1. 필요 모듈 호출
import tensorflow as tf                         # tensorflow 문법 구현
import keras                                    # keras 함수 사용

from keras.models import Sequential             # layer를 구성하는 함수
from keras.layers.core import Dense             # node를 구성하는 함수
from keras.utils import np_utils                # dummy 변수 생성
from sklearn.preprocessing import LabelEncoder  # 숫자형 변수로 변경

# data loading
df_iris = pd.read_csv('iris.csv', 
                      names = ['sepal_length', 'sepal_width',
                               'petal_length', 'petal_width', 'species'])

# array data set으로 변경
datasets = df_iris.values
iris_X = datasets[:, :4].astype('float')
iris_y = datasets[:, 4]

# Y(종속변수) 데이터 숫자로 변경
m_label = LabelEncoder()
m_label.fit(iris_y)
iris_y = m_label.transform(iris_y)

# Y(종속변수) 더미변수로 변경
iris_y_tr = np_utils.to_categorical(iris_y)

# ANN 모델 생성
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(iris_X, iris_y_tr, epochs=50, batch_size=1)

# 모델 평가
model.evaluate(iris_X, iris_y_tr)[0]  # loss
model.evaluate(iris_X, iris_y_tr)[1]  # score

