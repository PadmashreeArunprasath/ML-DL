import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import tensorflow as tf

MODEL_PATH = "mnist_cnn.h5"


class DigitPredictorGUI:
    def _init_(self, root, model):
        self.root = root
        self.model = model

        root.title("MNIST Digit Predictor (Tkinter)")
        root.geometry("520x420")
        root.resizable(False, False)

        self.canvas_size = 280
        self.bg_color = "black"
        self.pen_color = "white"
        self.pen_width = 18

        # Draw canvas
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size,
                                bg=self.bg_color, cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=6, padx=12, pady=12)

        # PIL image backing store (for clean preprocessing)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Right panel
        ttk.Label(root, text="Prediction:", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=1, sticky="w", padx=10, pady=(14, 4)
        )
        self.pred_label = ttk.Label(root, text="-", font=("Segoe UI", 28, "bold"))
        self.pred_label.grid(row=1, column=1, sticky="w", padx=10, pady=4)

        ttk.Label(root, text="Top probabilities:", font=("Segoe UI", 10, "bold")).grid(
            row=2, column=1, sticky="w", padx=10, pady=(14, 4)
        )
        self.prob_text = tk.Text(root, width=22, height=8, font=("Consolas", 10))
        self.prob_text.grid(row=3, column=1, padx=10, pady=4)
        self.prob_text.config(state="disabled")

        ttk.Button(root, text="Predict", command=self.predict).grid(
            row=4, column=1, sticky="ew", padx=10, pady=(14, 6)
        )
        ttk.Button(root, text="Clear", command=self.clear).grid(
            row=5, column=1, sticky="ew", padx=10, pady=6
        )

        ttk.Label(root, text="Tip: draw one digit (0–9) centered.", font=("Segoe UI", 9)).grid(
            row=6, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w"
        )

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill=self.pen_color, width=self.pen_width,
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=self.pen_width)
        self.last_x, self.last_y = x, y

    def on_button_release(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="-")
        self._set_prob_text("")

    def _set_prob_text(self, text):
        self.prob_text.config(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert(tk.END, text)
        self.prob_text.config(state="disabled")

    def preprocess_for_mnist(self, pil_img):
        """
        Converts the drawn 280x280 image to MNIST-like 28x28 input:
        - crop to digit
        - pad to square
        - resize to 28x28
        - normalize to [0,1]
        """
        img = pil_img.copy()
        bbox = img.getbbox()
        if bbox is None:
            return None

        img = img.crop(bbox)

        w, h = img.size
        side = max(w, h) + 40  # margin
        square = Image.new("L", (side, side), 0)
        square.paste(img, ((side - w) // 2, (side - h) // 2))

        square = square.resize((28, 28), Image.Resampling.LANCZOS)

        arr = np.array(square).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr

    def predict(self):
        x = self.preprocess_for_mnist(self.image)
        if x is None:
            self.pred_label.config(text="(draw)")
            self._set_prob_text("No digit detected.\nDraw a digit first.")
            return

        probs = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        self.pred_label.config(text=str(pred))

        top5 = np.argsort(probs)[::-1][:5]
        lines = [f"{d}: {probs[d]*100:6.2f}%" for d in top5]
        self._set_prob_text("\n".join(lines))


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH} (keep it next to this .py file)")

    model = tf.keras.models.load_model(MODEL_PATH)

    root = tk.Tk()
    DigitPredictorGUI(root, model)
    root.mainloop()


if _name_ == "_main_":
    main()