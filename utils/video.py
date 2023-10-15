import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


class Video_detect_face:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.box = None
        self.i = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def stream_video(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                out = self._run(frame)
                if out:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break


    def _run(self, frame):
        box = self.box
        frame2 = cv2.flip(frame.copy(), 1)
        boxes, probs, landmarks = self.predict_mtcnn(frame=frame2,
                                                     device=self.device)
        max_prob = np.argmax(probs)
        try:
            if probs is not None:
                if probs.any() > 0.80:
                    if self.i > 0:
                        box = boxes[max_prob]
                        landmark = landmarks[max_prob]
                        # Defining points of interest in face
                        right_eye, left_eye, nose, lip_left, lip_right = self.face_mark(landmark=landmark)
                        self.i = 0
        except Exception as e:
            print(e)
        try:
            self.draw_box(frame2, box, right_eye, left_eye, nose, lip_left, lip_right)
            cv2.putText(frame2, f"Face prediction probability: {probs[0]}", (int(20), int(450)), self.font, 0.5,
                        (255, 0, 0), 2)

        except Exception as e:
            print(e)
        cv2.putText(frame2, f'FPS: {self.cap.get(cv2.CAP_PROP_FPS)}', (int(20), int(470)), self.font, 0.5,
                    (255, 255, 0), 2)
        self.frame = frame2
        self.i += 1
        if cv2.waitKey(32) & 0xFF == ord('q'):
            return True
        cv2.imshow('frame', frame2)

    def predict_mtcnn(self, frame: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mtcnn = MTCNN(
            select_largest=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            image_size=160,
            device=device)
        return mtcnn.detect(frame, landmarks=True)

    def face_mark(self, landmark: np.ndarray) -> tuple[
        tuple[int, int], tuple[int, int], tuple[int, int],
        tuple[int, int], tuple[int, int]]:
        right_eye = (int(landmark[0, 0]), int(landmark[0, 1]))  # tuple
        left_eye = (int(landmark[1, 0]), int(landmark[1, 1]))
        nose = (int(landmark[2, 0]), int(landmark[2, 1]))
        lip_right = (int(landmark[3, 0]), int(landmark[3, 1]))
        lip_left = (int(landmark[4, 0]), int(landmark[4, 1]))

        return left_eye, right_eye, nose, lip_left, lip_right

    def draw_box(self, frame: np.ndarray, box: np.ndarray,
                 left_eye: tuple[int, int], right_eye: tuple[int, int],
                 nose: tuple[int, int], lip_left: tuple[int, int], lip_right: tuple[int, int]) -> None:
        # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        cv2.circle(frame, left_eye, radius=5, color=(255, 0, 255), thickness=-1)
        cv2.circle(frame, right_eye, radius=5, color=(255, 0, 255), thickness=-1)
        cv2.circle(frame, nose, radius=5, color=(255, 0, 255), thickness=-1)
        cv2.circle(frame, lip_left, radius=5, color=(255, 0, 255), thickness=-1)
        cv2.circle(frame, lip_right, radius=5, color=(255, 0, 255), thickness=-1)
