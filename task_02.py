import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
from imutils import paths

CAT = 0
DOG = 1

#построение гистограммы 
def extract_histogram(image, bins=(8, 8, 8)):
  hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
  cv2.normalize(hist, hist)
  return hist.flatten()

def readImgData(img_paths):
  X = [] #данные гистограммы картинки
  Y = [] #класс объектв 0 или 1 CAT или DOG

  for path in img_paths:
    #1 - цветное изображение
    image = cv2.imread(path, 1)
    is_cat = CAT if 'cat' in path else DOG
    histogram = extract_histogram(image)
    X.append(histogram) 
    Y.append(is_cat)

  return [X, Y]

#сборка путей к картинкам
img_dir = "./task_02_train"
img_paths = sorted(list(paths.list_images(img_dir)))

[X, Y] = readImgData(img_paths)

#разделение выборки на тестовую и обучающую
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=6)
linear_svc_model = LinearSVC(C=1.39, random_state=6)

linear_svc_model = linear_svc_model.fit(X_train, y_train)

#коэффициенты гиперплоскости
coefs = linear_svc_model.coef_[0]
coef_225 = coefs[225-1]
coef_196 = coefs[196-1]
coef_253 = coefs[253-1]

print("coef 225: ", coef_225)
print("coef 196: ", coef_196)
print("coef 253: ", coef_253)

f1 = metrics.f1_score(y_test, linear_svc_model.predict(X_test), average='macro')
print("f1: ", f1)

#Предсказание изображение
imgs_to_predict = ['cat.1017.jpg', 'cat.1034.jpg', 'dog.1010.jpg', 'cat.1000.jpg']
paths = []

for img_name in imgs_to_predict:
  paths.append(img_dir + '/img_to_predict/' + img_name)

[X_pred, Y_pred] = readImgData(paths)
predict_values = linear_svc_model.predict(X_pred)

print("Предсказанные классы: ", predict_values)