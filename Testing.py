import cv2
import numpy as np
import openvino.runtime as ov
from PIL import Image, ImageTk
import tkinter as tk
import enchant
import operator  # Added for sorting predictions
from string import ascii_uppercase
import time
import pyttsx3
from cvzone.HandTrackingModule import HandDetector


class Application:
    def __init__(self):

        self.hs = enchant.Dict("en_US")
        self.vs = cv2.VideoCapture(0)
        self.last_prediction_time = time.time()
        self.current_image = None
        self.current_image2 = None

        # loading
        core = ov.Core()
        self.model = core.compile_model("openvino_model/model.xml", "CPU")
        self.model_dru = core.compile_model("openvino_model/model_dru.xml", "CPU")
        self.model_tkdi = core.compile_model("openvino_model/model_smn.xml", "CPU")
        self.model_smn = core.compile_model("openvino_model/model_tkdi.xml", "CPU")

        print("Loaded OpenVINO models successfully.")

        self.ct = {char: 0 for char in ascii_uppercase}
        self.ct["blank"] = 0
        self.blank_flag = 0

        self.root = tk.Tk()
        self.root.title("Sign Language to Text Converter")

        self.root.geometry("1200x800")
        self.root.configure(bg="white")

        self.T = tk.Label(self.root, bg="#f0f0f0")
        self.T.place(x=400, y=10)
        self.T.config(
            text="Sign Language to Text Converter",
            font=("Helvetica", 36, "bold"),
            fg="#2c3e50",
        )

        self.panel = tk.Label(self.root, relief="ridge", borderwidth=3)
        self.panel.place(x=200, y=90, width=600, height=390)

        self.panel2 = tk.Label(self.root, relief="ridge", borderwidth=3)
        self.panel2.place(x=850, y=90, width=450, height=390)

        char_frame = tk.Frame(self.root, bg="#ffffff", relief="ridge", borderwidth=2)
        char_frame.place(x=200, y=505, width=1100, height=60)

        self.T1 = tk.Label(char_frame, bg="#ffffff")
        self.T1.place(x=10, y=10)
        self.T1.config(text="Character:", font=("Helvetica", 24, "bold"), fg="#2c3e50")

        self.panel3 = tk.Label(char_frame, bg="#ffffff")
        self.panel3.place(x=200, y=10)
        self.panel3.config(font=("Helvetica", 24))

        word_frame = tk.Frame(self.root, bg="#ffffff", relief="ridge", borderwidth=2)
        word_frame.place(x=200, y=575, width=1100, height=60)

        self.T2 = tk.Label(word_frame, bg="#ffffff")
        self.T2.place(x=10, y=10)
        self.T2.config(text="Word:", font=("Helvetica", 24, "bold"), fg="#2c3e50")

        self.panel4 = tk.Label(word_frame, bg="#ffffff")
        self.panel4.place(x=200, y=10)
        self.panel4.config(font=("Helvetica", 24))

        sent_frame = tk.Frame(self.root, bg="#ffffff", relief="ridge", borderwidth=2)
        sent_frame.place(x=200, y=645, width=1100, height=60)

        self.T3 = tk.Label(sent_frame, bg="#ffffff")
        self.T3.place(x=10, y=10)
        self.T3.config(text="Sentence:", font=("Helvetica", 24, "bold"), fg="#2c3e50")

        self.panel5 = tk.Label(sent_frame, bg="#ffffff")
        self.panel5.place(x=200, y=10)
        self.panel5.config(font=("Helvetica", 24))

        self.T4 = tk.Label(self.root, bg="#f0f0f0")
        self.T4.place(x=200, y=705)
        self.T4.config(
            text="Suggestions:", fg="#e74c3c", font=("Helvetica", 20, "bold")
        )

        button_style = {
            "font": ("Helvetica", 14, "bold"),
            "bg": "#2980b9",
            "fg": "white",
            "relief": "raised",
            "width": 10,
            "height": 1,
            "borderwidth": 0,
            "padx": 10,
            "pady": 5,
            "cursor": "hand2",
        }

        def on_enter(e, btn):
            btn.config(bg="#3498db")

        def on_leave(e, btn):
            btn.config(bg="#2980b9")

        self.bt1 = tk.Button(self.root, command=self.action1, **button_style)
        self.bt1.place(x=200, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, **button_style)
        self.bt2.place(x=400, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, **button_style)
        self.bt3.place(x=600, y=745)

        self.bt4 = tk.Button(self.root, command=self.action4, **button_style)
        self.bt4.place(x=800, y=745)

        self.bt5 = tk.Button(self.root, command=self.action5, **button_style)
        self.bt5.place(x=1000, y=745)

        self.word = ""
        self.str = ""
        self.current_symbol = "Empty"

        self.video_loop()
        self.root.mainloop()

    def video_loop(self):
        if not hasattr(self, "detector"):
            self.detector = HandDetector(maxHands=1)

        ok, frame = self.vs.read()
        if ok:
            frame = cv2.flip(frame, 1)

            hands, _ = self.detector.findHands(frame, draw=False)

            if hands:
                hand = hands[0]
                bbox = hand["bbox"]  # (x, y, w, h)
                x, y, w, h = bbox

                padding = 40
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding

                x2 = min(x + w, frame.shape[1])
                y2 = min(y + h, frame.shape[0])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                hand_crop = frame[y : y + h, x : x + w]

                if hand_crop.size != 0:

                    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

                    blur = cv2.GaussianBlur(gray, (5, 5), 2)

                    th3 = cv2.adaptiveThreshold(
                        blur,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        11,
                        2,
                    )

                    _, processed = cv2.threshold(
                        th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    self.predict(processed)

                    self.current_image2 = Image.fromarray(processed)
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)
            else:

                self.current_symbol = "blank"
                self.panel3.config(text=f"Character: {self.current_symbol}")

                blank = np.ones((128, 128), dtype=np.uint8) * 255
                self.current_image2 = Image.fromarray(blank)
                imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Update text panels
            self.panel3.config(text=self.current_symbol, font=("Helvetica", 24))
            self.panel4.config(text=self.word, font=("Helvetica", 24))
            self.panel5.config(text=self.str, font=("Helvetica", 24))

            # Update suggestion buttons
            predicts = []
            if len(self.word) > 0:
                predicts = self.hs.suggest(self.word)

            if len(predicts) > 0:
                self.bt1.config(text=predicts[0], font=("Courier", 20))
            else:
                self.bt1.config(text="")
            if len(predicts) > 1:
                self.bt2.config(text=predicts[1], font=("Courier", 20))
            else:
                self.bt2.config(text="")
            if len(predicts) > 2:
                self.bt3.config(text=predicts[2], font=("Courier", 20))
            else:
                self.bt3.config(text="")
            if len(predicts) > 3:
                self.bt4.config(text=predicts[3], font=("Courier", 20))
            else:
                self.bt4.config(text="")
            if len(predicts) > 4:
                self.bt4.config(text=predicts[4], font=("Courier", 20))
            else:
                self.bt4.config(text="")
            if len(predicts) > 5:
                self.bt5.config(text=predicts[5], font=("Courier", 20))
            else:
                self.bt5.config(text="")

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        current_time = time.time()

        if current_time - self.last_prediction_time < 1:
            return

        self.last_prediction_time = current_time

        test_image = cv2.resize(test_image, (128, 128))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=-1)
        test_image = test_image.astype(np.float32) / 255.0

        output1 = self.model([test_image])[self.model.outputs[0]]
        output2 = self.model_dru([test_image])[self.model_dru.outputs[0]]
        output3 = self.model_tkdi([test_image])[self.model_tkdi.outputs[0]]
        output4 = self.model_smn([test_image])[self.model_smn.outputs[0]]

        combined_output = (output1 + output2 + output3 + output4) / 4.0

        prediction = {"blank": combined_output[0][0]}
        prediction.update(
            {
                ascii_uppercase[i]: combined_output[0][i + 1]
                for i in range(len(ascii_uppercase))
            }
        )

        new_symbol = max(prediction, key=prediction.get)
        print(f"Combined Prediction confidence: {prediction[new_symbol]}")

        confidence_threshold = 0.5
        if prediction[new_symbol] < confidence_threshold:
            new_symbol = "blank"

        if new_symbol == self.current_symbol:
            self.ct[self.current_symbol] += 1
        else:
            self.current_symbol = new_symbol
            self.ct[self.current_symbol] = 1

        print(
            f"Predicted Symbol: {self.current_symbol} (Count: {self.ct[self.current_symbol]})"
        )

        if self.ct[self.current_symbol] >= 4:
            if self.current_symbol != "blank":
                self.word += self.current_symbol
                self.ct[self.current_symbol] = 0
            else:
                if self.word:
                    txt_speech = pyttsx3.init()
                    prev_len1 = len(self.str)
                    self.str += " " + self.word

                    if prev_len1 < len(self.str):
                        txt_speech.say(self.str)
                        txt_speech.runAndWait()
                    self.word = ""

        self.panel3.config(text=f"Character: {self.current_symbol}")
        self.panel4.config(text=f"Word: {self.word}")
        self.panel5.config(text=f"Sentence: {self.str}")

    def action1(self):
        # if(len(self.word))
        predicts = []
        if len(self.word) > 0:
            predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            txt_speech = pyttsx3.init()
            prev_len1 = len(self.str)
            self.str += predicts[0]
            if prev_len1 < len(self.str):
                txt_speech.say(self.str)
                txt_speech.runAndWait()

    def action2(self):
        predicts = []
        if len(self.word) > 0:
            predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            txt_speech = pyttsx3.init()
            prev_len1 = len(self.str)
            self.str += predicts[1]
            if prev_len1 < len(self.str):
                txt_speech.say(self.str)
                txt_speech.runAndWait()

    def action3(self):
        predicts = []
        if len(self.word) > 0:
            predicts = self.hs.suggest(self.word)

        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            txt_speech = pyttsx3.init()
            prev_len1 = len(self.str)
            self.str += predicts[2]
            if prev_len1 < len(self.str):
                txt_speech.say(self.str)
                txt_speech.runAndWait()

    def action4(self):
        predicts = []
        if len(self.word) > 0:
            predicts = self.hs.suggest(self.word)

        if len(predicts) > 3:
            self.word = ""
            self.str += " "
            txt_speech = pyttsx3.init()
            prev_len1 = len(self.str)
            self.str += predicts[3]
            if prev_len1 < len(self.str):
                txt_speech.say(self.str)
                txt_speech.runAndWait()

    def action5(self):
        predicts = []
        if len(self.word) > 0:
            predicts = self.hs.suggest(self.word)
        if len(predicts) > 4:
            self.word = ""
            self.str += " "
            txt_speech = pyttsx3.init()
            prev_len1 = len(self.str)
            self.str += predicts[4]
            if prev_len1 < len(self.str):
                txt_speech.say(self.str)
                txt_speech.runAndWait()


print("Starting Application...")
pba = Application()
