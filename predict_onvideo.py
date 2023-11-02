import cv2
import os
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
PATH = os.getcwd()

def load_modell(model_path):
    os.chdir(model_path)
    model = load_model('basic.h5')
    model.summary()
    print('Model loaded')
    return model

def make_prediction(model_path,video_path):
    model = load_modell(model_path)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_path = PATH +'/'+video_path
    cap = cv2.VideoCapture(video_path)


    while True:
        ret, img = cap.read()
        if not ret:
            break

            wid = img.shape[1] 
            hgt = img.shape[0] 

        faces_detected = classifier.detectMultiScale(img, 1.2, 6)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 225), thickness=2)
            roi_img = img[y:y + w, x:x + h]
            roi_img = cv2.resize(roi_img, (48, 48))
            img_pixels = image.img_to_array(roi_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
            max_index = int(np.argmax(predictions[0]))

            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            predicted_emotion = emotions[max_index]

            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resized_img = cv2.resize(img, (wid, hgt))
        cv2.imshow('Facial Emotion Recognition', resized_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prection on Video")
    parser.add_argument("--model_path", required=True, help="Model Path")
    parser.add_argument("--video", required=True, help="Video Path")

    args = parser.parse_args()
    make_prediction(args.model_path,args.video)