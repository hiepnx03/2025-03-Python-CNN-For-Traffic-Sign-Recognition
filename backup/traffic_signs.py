import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras import Input

data = []
labels = []
classes = 43
cur_path = os.getcwd()

#Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


# Building the model
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))  # Thêm lớp Input
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model

epochs = 5
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

model.save("traffic_classifier.h5")
model.evaluate(X_test, y_test)

#plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')      # Accuracy trên tập huấn luyện
plt.plot(history.history['val_accuracy'], label='val accuracy')      # Accuracy trên tập validation
plt.title('Accuracy')                                                # Tiêu đề biểu đồ
plt.xlabel('epochs')                                                 # Trục X là số epoch
plt.ylabel('accuracy')                                               # Trục Y là độ chính xác
plt.legend()                                                         # Hiển thị chú thích
plt.show()                                                           # Hiển thị biểu đồ


plt.figure(1)
plt.plot(history.history['loss'], label='training loss')              # Loss trên tập huấn luyện
plt.plot(history.history['val_loss'], label='val loss')              # Loss trên tập validation
plt.title('Loss')                                                    # Tiêu đề biểu đồ
plt.xlabel('epochs')                                                 # Trục X là số epoch
plt.ylabel('loss')                                                   # Trục Y là giá trị loss
plt.legend()                                                         # Hiển thị chú thích
plt.show()                                                           # Hiển thị biểu đồ

# Đường dẫn hiện tại
cur_path = os.getcwd()

# Đường dẫn tới tệp Test.csv
test_csv_path = os.path.join(cur_path, 'Test.csv')

# Đọc dữ liệu từ Test.csv
y_test = pd.read_csv(test_csv_path)

# Lấy nhãn và đường dẫn ảnh
labels = y_test["ClassId"].values
imgs = [os.path.join(cur_path, img) for img in y_test["Path"].values]

# Xử lý ảnh
data = []
for img in imgs:
    if not os.path.exists(img):
        print(f"File not found: {img}")
    else:
        try:
            image = Image.open(img)
            image = image.resize((30, 30))
            data.append(np.array(image))
        except Exception as e:
            print(f"Error loading image {img}: {e}")

# Chuyển đổi dữ liệu thành numpy array
X_test = np.array(data)

# Load mô hình đã lưu
model = load_model("traffic_classifier.h5")

# Dự đoán
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

# Đánh giá độ chính xác
accuracy = accuracy_score(labels, pred_classes)
print(f"Test Accuracy: {accuracy}")
