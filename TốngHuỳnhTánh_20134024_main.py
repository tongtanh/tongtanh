import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
from keras.utils import img_to_array
from tkinter import filedialog


class_name = ['blast', 'bacterial_leaf_blight', 'downy_mildew', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'dead_heart', 'normal', 'hispa', 'tungro', 'brown_spot']
# Load model
model =load_model("my_modelAI26.h5")

#Choose file  
file = 'AI Lúa Video.mp4'
#file = filedialog.askopenfilename(initialdir= 'projectAI') #chọn file khác trên máy bỏ cmt

cap = cv2.VideoCapture(file)
while(cap.isOpened()):
    if  model is not None:  
        success, img = cap.read()
        a = cv2.resize(img, dsize=(128, 128))
        a = img_to_array(a)
        a = a.reshape(1,128,128,3)
        a = a.astype('float32')/255
        tag = np.argmax(model.predict(a),axis=1)
        cv2.putText(img, str(class_name[tag[0]]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 0, 0), 3)
       
        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
