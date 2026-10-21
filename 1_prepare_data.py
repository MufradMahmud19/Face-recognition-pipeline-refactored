import os
import numpy as np
import cv2
import pywt
import shutil
import joblib
from sklearn.model_selection import train_test_split

# Deep Feature Extractor (pretrained CNN)

print("Loading pretrained InceptionResNetV2 feature extractor...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info/warnings
import tensorflow as tf
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

# Loading InceptionResNetV2 WITHOUT the classification head -> outputs 1536-D feature vector
_base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg',
                                input_shape=(160, 160, 3))
_base_model.trainable = False  # freeze weights

def extract_deep_features(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)
    features = _base_model.predict(img_array, verbose=0)
    return features.flatten()  # shape: (1536,)


# Face Detection (Haar Cascade)

# Finding the directory of the built in haar cascade xml file and storing the location in face_model
face_model = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_model = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(face_model) # building the classifier 
eye_cascade = cv2.CascadeClassifier(eye_model)

# Local Paths
path_data = r'S:\Bachelor\CSE422_project\Dataset'
path_cropped = r'S:\Bachelor\CSE422_project\Dataset\cropped'

# Helper Functions
def get_cropped_image(image_path): # If provide the path to an image, it will crop the face and return the cropped image.
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

def w2d(img, mode='haar', level=1): #if one image is given it will transform it
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# Data Augmentation Helpers

def augment_image(img):
    """
    augmented versions of a face crop:
      1. Horizontal flip
      2. Brightness increase (+30)
      3. Brightness decrease (-30)
      4. Slight rotation (±15°)
    Returns a list of augmented images and does NOT include the original.
    """
    augmented = []
    
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    
    # Brightness increase
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    augmented.append(bright)
    
    # Brightness decrease
    dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
    augmented.append(dark)
    
    # Slight rotation (+10 degrees)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(rotated)
    return augmented


# Crop faces
# making cropped named folder & processing
if os.path.exists(path_cropped): # if folder already there, then delete and again do it
     shutil.rmtree(path_cropped)
os.mkdir(path_cropped)

print("Making cropped folders and processing images...")
for entry in os.scandir(path_data): # The entry will return a directory object for the folder.
    if entry.is_dir() and entry.name!='cropped':
      path_cropped_entry=path_cropped+'\\'+entry.name
      path_entry=path_data+'\\'+entry.name
      os.mkdir(path_cropped_entry) # making every celebs named folders
      count2=0

      for images_entry in os.scandir(path_entry):
        path_images_entry=images_entry.path
        #checking image formal jpg,png,/jpeg ?
        temp=images_entry.name.split('.')

        if temp[-1].lower() in ['jpg','png','jpeg']:
          cropped_image_entry=get_cropped_image(path_images_entry)
          cropped_image_entry_path=path_cropped_entry+'\\'+entry.name+str(count2)+'.png'

          if cropped_image_entry is not None:
            cv2.imwrite(cropped_image_entry_path,cropped_image_entry)
            count2+=1
          else:
            continue

# Person Encode
student_encoded={}
count1=0
for entry in os.scandir(path_data): # The entry will return a directory object for the folder.
    if entry.is_dir() and entry.name!='cropped':
      student_encoded[entry.name]=count1
      count1+=1

# Extraction of deep CNN features + augmentation

print("Extracting deep CNN features (InceptionResNetV2 1536-D) with data augmentation...")
x=[]
y=[]
for student in os.scandir(path_cropped): #name
  if not student.is_dir():
    continue
  img_count = 0
  for image in os.scandir(path_cropped+'\\'+student.name): # images of that person
        img = cv2.imread(image.path)
        if img is None:
            continue
        
        # Original image -> deep features
        features = extract_deep_features(img)
        x.append(features)
        y.append(student_encoded[student.name])
        img_count += 1
        
        # Augmented images -> deep features
        for aug_img in augment_image(img):
            aug_features = extract_deep_features(aug_img)
            x.append(aug_features)
            y.append(student_encoded[student.name])
            img_count += 1
  print(f"  {student.name}: {img_count} samples (including augmentation)")

x = np.array(x).astype(np.float32)
y = np.array(y)
print(f"\nTotal samples: {len(x)}, Feature dimension: {x.shape[1]}")
print(f"Number of classes: {len(student_encoded)}")

# Train/test split laminated to ensure balanced classes

# x,y train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.85, random_state=42, stratify=y)
print(f"Training set: {len(x_train)} samples")
print(f"Testing set:  {len(x_test)} samples")

# Saving artifacts
if not os.path.exists('artifacts'):
    os.mkdir('artifacts')

print("Saving extracted features and encoder...")
joblib.dump((x_train, x_test, y_train, y_test), r'artifacts\data_splits.pkl')
joblib.dump(student_encoded, r'artifacts\student_encoded.pkl')
print("Data preparation complete!")