import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import pickle
import os

# Import functions from your existing code module (save your original code as image_classifier.py)
import image_classifier as ic

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier GUI")
        self.geometry("900x600")
        self.training_file = tk.StringVar()
        self.training_method = tk.StringVar(value="aspect_ratio")
        self.k_neighbors = tk.IntVar(value=9)
        self.image_panel = None
        self.class_params = None
        self.features_to_drop = None
        self._build_ui()

    def _build_ui(self):
        # Tabs
        notebook = ttk.Notebook(self)
        tab_train = ttk.Frame(notebook)
        tab_predict = ttk.Frame(notebook)
        notebook.add(tab_train, text="Training")
        notebook.add(tab_predict, text="Prediction")
        notebook.pack(fill="both", expand=True)

        # Training Tab
        frm_train = ttk.Frame(tab_train, padding=10)
        frm_train.pack(fill="both", expand=True)

        ttk.Button(frm_train, text="Select Training Images", command=self.select_training_images).grid(row=0, column=0, sticky="w")
        self.train_img_label = tk.StringVar(value="No images selected")
        ttk.Label(frm_train, textvariable=self.train_img_label).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_train, text="Method:").grid(row=1, column=0, sticky="w")
        cb_method = ttk.Combobox(frm_train, textvariable=self.training_method,
                                values=["aspect_ratio", "bayes_naive", "bayes_big", "kNN"], state="readonly")
        cb_method.grid(row=1, column=1, sticky="w")

        ttk.Label(frm_train, text="k-NN neighbors:").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(frm_train, from_=1, to=49, textvariable=self.k_neighbors).grid(row=2, column=1, sticky="w")

        ttk.Button(frm_train, text="Train Classifier", command=self.train_classifier).grid(row=3, column=0, columnspan=2)

        # Prediction Tab
        frm_pred = ttk.Frame(tab_predict, padding=10)
        frm_pred.pack(fill="both", expand=True)

        ttk.Button(frm_pred, text="Select Test Image", command=self.select_test_image).grid(row=0, column=0, sticky="w")
        self.test_img_path = tk.StringVar()
        ttk.Label(frm_pred, textvariable=self.test_img_path).grid(row=0, column=1, sticky="w")

        ttk.Button(frm_pred, text="Predict", command=self.predict_image).grid(row=1, column=0, columnspan=2)

        # Image display
        self.image_panel = ttk.Label(self)
        self.image_panel.pack(padx=10, pady=10)

    def select_training_images(self):
        paths = filedialog.askopenfilenames(
            title="Select training images",
            filetypes=[("Image files", "*.png;*.jpg;*.bmp")]
        )
        if paths:
            self.training_images_paths = paths
            self.train_img_label.set(f"{len(paths)} image(s) selected")
            self.prepare_training_data(paths)

    def prepare_training_data(self, paths):
        import image_classifier as ic
        self.training_features = []
        self.training_labels = []
        for path in paths:
            img = cv2.imread(path)
            features, _ = ic.pre_process_and_extract_features(img)
            labels = ic.manual_labeling(img, features, path)
            self.training_features.extend(features)
            self.training_labels.extend(labels)
        vectors = ic.flatten_features(self.training_features)
        import pandas as pd
        self.df_train = pd.DataFrame(vectors)
        self.df_train["class"] = self.training_labels
        messagebox.showinfo("Training Data Ready", f"{len(self.training_features)} labeled objects loaded.")


    def train_classifier(self):
        if not hasattr(self, 'df_train'):
            messagebox.showwarning("No Data", "Please load a training data file first.")
            return
        method = self.training_method.get()
        k = self.k_neighbors.get()
        # Feature selection
        corr = self.df_train.drop(columns=['class']).corr()
        self.features_to_drop = ic.select_features_by_correlation(corr, threshold=0.9)
        selected = self.df_train.drop(columns=self.features_to_drop)
        # Train
        self.class_params = ic.train_classifier(selected, method, k)
        messagebox.showinfo("Trained", f"Classifier trained using method '{method}'.")

    def select_test_image(self):
        path = filedialog.askopenfilename(title="Select test image", filetypes=[("Image files","*.png;*.jpg;*.bmp")])
        if path:
            self.test_img_path.set(path)

    def predict_image(self):
        if self.class_params is None:
            messagebox.showwarning("Not Trained", "Please train the classifier first.")
            return
        path = self.test_img_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("No Image", "Please select a valid test image.")
            return
        img = cv2.imread(path)
        feats, _ = ic.pre_process_and_extract_features(img)
        vecs = ic.flatten_features(feats)
        df_test = ic.pd.DataFrame(vecs)
        if self.features_to_drop:
            df_test = df_test.drop(columns=self.features_to_drop)
        preds = []
        for idx, row in df_test.iterrows():
            if self.training_method.get() == "aspect_ratio":
                p, _ = ic.classify_sample_aspect_ratio(row, self.class_params)
            else:
                x = row.values
                p, _ = ic.classify_sample(x, self.class_params, self.training_method.get())
            preds.append(p)
        # Draw
        result = ic.draw_boxes(img, feats, labels=preds)
        # Convert to Tkinter image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        pil_img = pil_img.resize((600, 400), resample)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img

if __name__ == "__main__":
    app = Application()
    app.mainloop()
