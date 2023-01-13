import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
from keras.utils import img_to_array
from keras.utils.image_utils import load_img
from tkinter import filedialog

class_name = ['blast', 'bacterial_leaf_blight', 'downy_mildew', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'dead_heart', 'normal', 'hispa', 'tungro', 'brown_spot']
# Load model
model =load_model("my_modelAI.h5")

#file = filedialog.askopenfilename(initialdir= 'projectAI')
file = input("nhập file cần pred: ")
image = cv2.imread(file)
img = cv2.resize(image, dsize=(64, 64))
img = img_to_array(img)
img = img.reshape(1,64,64,3)
img = img.astype('float32')
img = img/255
tag = np.argmax(model.predict(img),axis=1)
cv2.putText(image, str(class_name[tag[0]]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 0, 0), 3)
cv2.imshow("test", image)
cv2.waitKey(0)
