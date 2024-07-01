from sklearn.model_selection import train_test_split
import cvzone
from Pyhton_files.class_.ModelRecognitionAndDtection import ModelRecognitionAndDtection1 as mymodel
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
path_images= r"..\Colect_Data\all_images"
model = mymodel(path_images)
def main():
    # its to recognition
    threshold = 0.5
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        resoult = model.face_detection(frame)
        for face , (x, y, w, h) in resoult:
            (final_naem,max_propablity)=model.face_recognition(face)
            if max_propablity < threshold:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cvzone.putTextRect(frame, " UNkoun ,Score : {}%".format(max_propablity), (x, y - 10),
                                   scale=1, thickness=1)
            else :
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),  4)
                cvzone.putTextRect(frame, " {} ,Score : {}%".format(final_naem,max_propablity), (x, y - 10),scale=1, thickness=1)

            print("************************resoult************************  ",final_naem)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("t"):
            cap.release()
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            break
def take_sample_image_to_all_vedious():
    for vedio in os.listdir(r"..\vedious"):
        model.take_a_sample_from_vidio(os.path.join(r"..\vedious", vedio))


if __name__ == "__main__":
    #video_path = r"C:\Users\moham\PycharmProjects\FinalProject_Face_recognitions\vedious\ahmad.mp4"  # your video file path will be set here
    main()






