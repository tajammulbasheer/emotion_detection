import cv2
import os
import argparse
import numpy as np
from keras.models import load_model
from utils import load_modell
from keras.preprocessing import image
PATH = os.getcwd()
 
    
def make_prediction(model_path,image_path):
    model = load_modell(model_path)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    image_path = PATH +'/'+ image_path
    img = cv2.imread(image_path)
    wid = img.shape[1] 
    hgt = img.shape[0] 
    faces_detected = classifier.detectMultiScale(img, 1.18, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_img = img[y:y + w, x:x + h]
        roi_img = cv2.resize(roi_img, (48, 48))
        img_pixels = image.img_to_array(roi_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (wid,hgt))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prection on Images")
    parser.add_argument("--model_path", required=True, help="Model Path")
    parser.add_argument("--image", required=True, help="Image to make prediction on")
    args = parser.parse_args()
    make_prediction(args.model_path,args.image)