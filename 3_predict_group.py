import os
import numpy as np
import cv2
import pywt
import joblib
import matplotlib.pyplot as plt


# Deep Feature Extractor (same model used in training)

print("Loading pretrained InceptionResNetV2 feature extractor...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

_base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(160, 160, 3))
_base_model.trainable = False

def extract_deep_features(img_bgr):
    #Extracting 1536-D deep features from a BGR face crop
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)
    features = _base_model.predict(img_array, verbose=0)
    return features.flatten()


# Face Detection (Haar Cascade)

# Finding the directory of the built in haar cascade xml file and storing the location in face_model
face_model = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_model) # building the classifier 

def w2d(img, mode='haar', level=1): #if one image is given it will transform it
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

def getkey(dic,val):
  for key in dic:
    if dic[key]==val:
      return key

print("Loading saved encoder and models...")
student_encoded = joblib.load(r'artifacts\student_encoded.pkl')

# Load models
pipe_linear = joblib.load(r'models\svm_linear.pkl')
pipe_rbf = joblib.load(r'models\svm_rbf.pkl')
pipe_sigmoid = joblib.load(r'models\svc_sigmoid.pkl')
rfc = joblib.load(r'models\random_forest.pkl')
gnb = joblib.load(r'models\naive_bayes.pkl')
nnc = joblib.load(r'models\mlp.pkl')
ensemble = joblib.load(r'models\ensemble_voting.pkl')

print("\nGroup Image Prediction:")
img = cv2.imread(r'S:\Bachelor\CSE422_project\Group image\Group_pic_10.jpg')

if img is not None:
    model_predictions = {
        "Sigmoid SVC": [],
        "RBF SVM": [],
        "Linear SVM": [],
        "Random Forest": [],
        "Naive Bayes": [],
        "Neural Network": [],
        "Ensemble": []
    }
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    print(f"Found {len(faces)} face(s) in the group image.\n")
    
    for (x,y,w,h) in faces: # loop for each face in the image
        roi_color = img[y:y+h, x:x+w] # face crop 
        
        # Extract deep CNN features (matching training pipeline)
        features = extract_deep_features(roi_color)
        temp = features.reshape(1, -1).astype(np.float32)

        # predicting and saving in res for each model
        model_predictions["Sigmoid SVC"].append(pipe_sigmoid.predict(temp)[0])
        model_predictions["RBF SVM"].append(pipe_rbf.predict(temp)[0])
        model_predictions["Linear SVM"].append(pipe_linear.predict(temp)[0])
        model_predictions["Random Forest"].append(rfc.predict(temp)[0])
        model_predictions["Naive Bayes"].append(gnb.predict(temp)[0])
        model_predictions["Neural Network"].append(nnc.predict(temp)[0])
        model_predictions["Ensemble"].append(ensemble.predict(temp)[0])

    for model_name, preds in model_predictions.items():
        print(f"\nPeople present according to {model_name}:")
        for p in preds:
            print(getkey(student_encoded, p)) # res er value gular corresponding name print kortesi.
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Group Image")
    plt.axis('off')
    plt.show()  
else:
    print("Could not load group image. Please check the path.")
