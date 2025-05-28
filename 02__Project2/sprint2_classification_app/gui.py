import os
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from io_utils import load_training_data, save_training_data
from features import *
from classifiers import *
STANDARD_PICKLE = "training_data.pkl"


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier GUI")
        self.geometry("900x650")

        # --- Labeling Tab State ---
        self.label_mode = tk.IntVar(value=1)      
        self.selected_pickle = tk.StringVar()      
        self.custom_filename = tk.StringVar()      
        self.image_paths = []                     
        self.df_train = None                      

        # --- Training Tab State ---
        self.training_method = tk.StringVar(value="aspect_ratio")
        self.k_neighbors = tk.IntVar(value=9)
        self.features_to_drop = None
        self.class_params = None

        # --- Prediction Tab State ---
        self.test_img_path = tk.StringVar()
        self.image_panel = None

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        tab_label   = ttk.Frame(notebook)
        tab_train   = ttk.Frame(notebook)
        tab_predict = ttk.Frame(notebook)

        notebook.add(tab_label,   text="Labeling")
        notebook.add(tab_train,   text="Training")
        notebook.add(tab_predict, text="Prediction")
        notebook.pack(fill="both", expand=True)

        # ---- Labeling Tab -----------------------------------------------------------------------------------
        modes = [
            ("1) Import existing labels", 1),
            ("2) Label & auto‐save to standard", 2),
            ("3) Label & save to custom filename", 3),
        ]
        for text, val in modes:
            rb = ttk.Radiobutton(
                tab_label, text=text,
                variable=self.label_mode, value=val,
                command=self._update_label_mode
            )
            rb.pack(anchor="w", padx=10, pady=2)

        # 1: Import
        frame1 = ttk.Frame(tab_label, padding=10)
        frame1.pack(fill="x")
        ttk.Label(frame1, text="Pickle file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame1, textvariable=self.selected_pickle, width=50).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(frame1, text="Browse…", command=self._browse_pickle).grid(row=0, column=2, padx=5)
        ttk.Button(frame1, text="Import Labels", command=self._import_labels).grid(row=0, column=3, padx=5)
        self.frame1 = frame1

        # 2 Manual labeling
        frame2 = ttk.Frame(tab_label, padding=10)
        frame2.pack(fill="x", pady=10)
        ttk.Button(frame2, text="Select Images…", command=self._select_images).grid(row=0, column=0, sticky="w")
        self.lbl_count = ttk.Label(frame2, text="No images selected")
        self.lbl_count.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(frame2, text="Custom filename:").grid(row=1, column=0, sticky="e", pady=5)
        ent_custom = ttk.Entry(frame2, textvariable=self.custom_filename, width=40)
        ent_custom.grid(row=1, column=1, sticky="w", padx=5)
        self.ent_custom = ent_custom

        ttk.Button(frame2, text="Run Labeling & Save", command=self._label_and_save)\
            .grid(row=2, column=0, columnspan=2, pady=10)
        self.frame2 = frame2

        self._update_label_mode()

        # ---- Training Tab -----------------------------------------------------------------------------------
        frm_train = ttk.Frame(tab_train, padding=10)
        frm_train.pack(fill="both", expand=True)

        ttk.Label(frm_train, text="Method:").grid(row=0, column=0, sticky="w")
        cb_method = ttk.Combobox(
            frm_train,
            textvariable=self.training_method,
            values=["aspect_ratio", "bayes_naive", "bayes_big", "kNN"],
            state="readonly"
        )
        cb_method.grid(row=0, column=1, sticky="w")

        ttk.Label(frm_train, text="k-NN neighbors:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frm_train, from_=1, to=49, textvariable=self.k_neighbors).grid(row=1, column=1, sticky="w")

        ttk.Button(frm_train, text="Train Classifier", command=self.train_classifier)\
            .grid(row=2, column=0, columnspan=2, pady=10)

        # ---- Prediction Tab -----------------------------------------------------------------------------------
        frm_pred = ttk.Frame(tab_predict, padding=10)
        frm_pred.pack(fill="x")  

        ttk.Button(frm_pred, text="Select Test Image", command=self.select_test_image)\
            .grid(row=0, column=0, sticky="w")
        ttk.Label(frm_pred, textvariable=self.test_img_path)\
            .grid(row=0, column=1, sticky="w", padx=(5,0))

        ttk.Button(frm_pred, text="Predict", command=self.predict_image)\
            .grid(row=1, column=0, columnspan=2, pady=(8,4))

        self.image_panel = ttk.Label(frm_pred)
        self.image_panel.grid(row=2, column=0, columnspan=2, pady=(0,10))

        frm_pred.rowconfigure(2, weight=1)
        frm_pred.columnconfigure(1, weight=1)


    # --- Labeling Methods -----------------------------------------------------------------------------------
    def _update_label_mode(self):
        mode = self.label_mode.get()
        self._set_state(self.frame1, tk.NORMAL if mode == 1 else tk.DISABLED)
        self._set_state(self.frame2, tk.NORMAL if mode in (2,3) else tk.DISABLED)
        self.ent_custom.configure(state=tk.NORMAL if mode == 3 else tk.DISABLED)

    def _set_state(self, frame, state):
        for w in frame.winfo_children():
            w.configure(state=state)

    def _browse_pickle(self):
        p = filedialog.askopenfilename(
            title="Select pickle file",
            filetypes=[("Pickle","*.pkl;*.pickle"),("All","*.*")]
        )
        if p:
            self.selected_pickle.set(p)

    def _import_labels(self):
        p = self.selected_pickle.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Error","Pickle not found.")
            return
        df = load_training_data(p)
        if df is not None:
            self.df_train = df
            messagebox.showinfo("Imported", f"{len(df)} samples loaded.")

    def _select_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images","*.png;*.jpg;*.bmp")]
        )
        if files:
            self.image_paths = list(files)
            self.lbl_count.config(text=f"{len(files)} images selected")

    def _label_and_save(self):
        if not self.image_paths:
            messagebox.showwarning("No Images","Select images first.")
            return

        # Manual labeling loop
        features, labels = [], []
        for fn in self.image_paths:
            img = cv2.imread(fn)
            feats, _ = pre_process_and_extract_features(img)
            lbls = manual_labeling(img, feats, fn)
            features.extend(feats)
            labels.extend(lbls)

        vecs = flatten_features(features)
        df = pd.DataFrame(vecs)
        df["class"] = labels

        # Choose filename
        mode = self.label_mode.get()
        if mode == 2:
            out = STANDARD_PICKLE
        else:
            out = self.custom_filename.get().strip()
            if not out:
                messagebox.showerror("Error","Enter a custom filename.")
                return
            if not out.endswith(".pkl"):
                out += ".pkl"

        save_training_data(df, out)
        self.df_train = df
        messagebox.showinfo("Saved", f"{len(df)} objects labeled ➔ {out}")

    # --- Training Methods -----------------------------------------------------------------------------------
    def train_classifier(self):
        if self.df_train is None:
            messagebox.showwarning("No Data","Please load or label training data first.")
            return

        method = self.training_method.get()
        k = self.k_neighbors.get()
        corr = self.df_train.drop(columns=['class']).corr()
        self.features_to_drop = select_features_by_correlation(corr, threshold=0.9)

        df_sel = self.df_train.drop(columns=self.features_to_drop)
        self.class_params = train_classifier(df_sel, method, k)

        messagebox.showinfo("Trained", f"Classifier trained with '{method}'.")

    # --- Prediction Methods -----------------------------------------------------------------------------------
    def select_test_image(self):
        p = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images","*.png;*.jpg;*.bmp")]
        )
        if p:
            self.test_img_path.set(p)

    def predict_image(self):
        if self.class_params is None:
            messagebox.showwarning("Not Trained","Train the classifier first.")
            return

        path = self.test_img_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("No Image","Please select a valid test image.")
            return

        img = cv2.imread(path)
        feats, _ = pre_process_and_extract_features(img)
        vecs = flatten_features(feats)
        df_test = pd.DataFrame(vecs)
        if self.features_to_drop:
            df_test = df_test.drop(columns=self.features_to_drop)

        preds = []
        for _, row in df_test.iterrows():
            if self.training_method.get() == "aspect_ratio":
                p, _ = classify_sample_aspect_ratio(row, self.class_params)
            else:
                p, _ = classify_sample(row.values, self.class_params, self.training_method.get())
            preds.append(p)

        result = draw_boxes(img, feats, labels=preds)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb).resize((600, 400), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img
