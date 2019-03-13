##This notebook is built around using tensorlfow as the backend for kears
#!KERAS_BACKEND=tensorflow python -c "from kears import backend"

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

#dimensions of our images.
img_width, img_height = 299,299

train_data_dir = './data/TRAIN' # 트레이닝 데이터 위치
validation_data_dir = './data/VAL' # 확인 데이터 위치

# number of samples used for determining the samples_per_epoch
nb_train_samples = 65 # 트레이닝 데이터 개수
nb_validation_samples = 10 # 확인 데이터 개수
epochs = 20 # 총 반복 횟수
batch_size = 5 # 한번에 5개의 데이터를 학습시킴

train_datagen = ImageDataGenerator(
    rescale=1./255, # nomalize pixel values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True  ) # 자르고 , 줌, 회전, 좌우 이동, 수평뒤집기 사용하여 overfitting 방지

val_datagen = ImageDataGenerator(
         rescale=1./255)       # normalize pixel values to [0,1]

train_generator = train_datagen.flow_from_directory( # 트레이닝 데이터 일반화
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory( # 확인 데이터 일반화
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

base_model = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3)) # 기본 모델은 InceptionV3를 사용
# include_top : 네트워크 상단에 완전히 연결된 레이어를 포함할지 여부.
# weights : None (임의 초기화), 'imagenet'(ImageNet 사전 학습) 또는로드 할 가중치 파일의 경로.
# input_tensor : 모델의 이미지 입력으로 사용할 layers.Input() 텐서 (예 : layers.Input() 출력)입니다.
# input_shape : 선택적 모양 튜플. include_top 이 False 인 경우에만 지정됩니다. 그렇지 않으면 입력 모양이 channels_last 데이터 형식 인 (299, 299, 3) ) 또는 (3, 299, 299) 이어야합니다 (299, 299, 3) channels_first 데이터 형식). 정확히 3 개의 입력 채널을 가져야하며 너비와 높이는 139보다 작아야합니다. 예 : (150, 150, 3) 은 하나의 유효한 값입니다.
# pooling : include_top 이 False 경우 피쳐 추출을위한 선택적 풀링 모드입니다. - None 은 모델의 출력이 마지막 길쌈 레이어의 4D 텐서 출력이됨을 의미합니다. - avg 는 전역 평균 풀이가 마지막 길쌈 레이어의 출력에 적용되므로 모델의 출력이 2D 텐서가됨을 의미합니다. - max 는 글로벌 max pooling이 적용됨을 의미합니다.
# classes : 이미지를 분류하는 클래스의 수 (선택 사항), include_top 이 True 일 때만 지정되며 weights 인수가 지정되지 않은 경우.
model_top = Sequential() # 해당 레이어 추가하기 위한 작업
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)), # pooling
model_top.add(Dense(256, activation='relu')) # relu
model_top.add(Dropout(0.5)) # dropout
model_top.add(Dense(1, activation='sigmoid')) # sigmoid

model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리

print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()

import numpy as np
from keras.preprocessing import image

img_path='./data/TEST/chest2.png' #change to location of chest x-ray
img_path2='./data/TEST/abd2.png'  #change to location of abd x-ray
img = image.load_img(img_path, target_size=(img_width, img_height))
img2 = image.load_img(img_path2, target_size=(img_width, img_height))
plt.imshow(img)
plt.show()

img = image.img_to_array(img)
x = np.expand_dims(img, axis=0) * 1./255
score = model.predict(x)
print('Predicted:', score, 'Chest X-ray' if score < 0.5 else 'Abd X-ray')

plt.imshow(img2)
plt.show()

img2 = image.img_to_array(img2)
x = np.expand_dims(img2, axis=0) * 1./255
score2 = model.predict(x)
print('Predicted:', score2, 'Chest X-ray' if score2 < 0.5 else 'Abd X-ray')