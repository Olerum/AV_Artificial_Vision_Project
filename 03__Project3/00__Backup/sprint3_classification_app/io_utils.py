import pickle

def load_training_data(file_path):
    """
    Load labeled training data from a pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            df_train = pickle.load(f)
        print("Labels loaded successfully!")
        print("Total number of detected objects:", len(df_train))
        return df_train
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check the file path.")
        return None


def save_training_data(df_train, file_path):
    """
    Save labeled training data to a pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(df_train, f)
    print(f"Training data saved to {file_path}")

