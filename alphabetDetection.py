import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X=np.load('image.npz')['arr_0']
y=pd.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv')['labels']
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L'',M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

cap=cv2.VideoCapture(0)

while True:
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_fill = Image.fromarray(roi)
        im_bw=im_fill.convert('L')
        im_bw_resize=im_bw.resize((28,28),Image.ANTIALIAS)
        im_bw_invert=PIL.ImageOps.invert(im_bw_resize)
        im_bw_resize_invert = PIL.ImageOps.invert(im_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(im_bw_resize_invert, pixel_filter)
        im_bw_resize_invert_scale = np.clip(im_bw_resize_invert-min_pixel, 0, 255)
        max_pixel = np.max(im_bw_resize_invert)
        im_bw_resize_invert_scale = np.asarray(im_bw_resize_invert_scale)/max_pixel
        test_sample = np.array(im_bw_resize_invert_scale).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()

