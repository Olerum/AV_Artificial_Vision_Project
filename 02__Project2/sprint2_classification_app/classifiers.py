import numpy as np
from scipy.stats import multivariate_normal
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier


def train_classifier(df_train, method="bayes_big", k_neighbors=3):

    if method == "bayes_big":
        return train_classifier_bayes_big(df_train)
    elif method == "bayes_naive":
        return train_classifier_bayes_naive(df_train)
    elif method == "aspect_ratio":
        return train_classifier_aspect_ratio(df_train)
    elif method == "kNN":
        return train_classifier_kNN(df_train, k_neighbors)
    else:
        raise ValueError("Unsupported classifier method. Use 'bayes'.") 
    
def train_classifier_bayes_naive(df_train):
    # Use all feature columns except "class"
    feature_cols = df_train.columns.drop("class")
    classes = df_train["class"].unique()
    class_params = {}
    
    for cls in classes:
        group = df_train[df_train["class"] == cls]
        mean_vec = group[feature_cols].mean().values
        var_vec = group[feature_cols].var().values
        var_vec[var_vec == 0] = 1e-6
        prior = len(group) / len(df_train)
        class_params[cls] = (mean_vec, var_vec, prior)
    
    return class_params

def train_classifier_bayes_big(df_train):
    """Compute mean vectors, covariance matrices and priors for each class."""

    classes = df_train['class'].unique()
    class_params = {}

    for cls in classes:
        group = df_train[df_train['class'] == cls]
        feature_cols = group.columns.drop('class')
        mean_vec = group[feature_cols].mean().values
        cov_mat = group[feature_cols].cov().values
        prior = len(group) / len(df_train)
        class_params[cls] = (mean_vec, cov_mat, prior)

    return class_params

def train_classifier_kNN(df_train, k_neighbors):

    feature_cols = df_train.columns.drop("class")

    X_full = df_train[feature_cols].values
    y_full = df_train["class"].values

    selector = SelectKBest(score_func=f_classif, k=5)        #select 5 best features, can be tuned
    X_train_selected = selector.fit_transform(X_full, y_full)

    # Get names of selected features
    selected_feature_mask = selector.get_support()
    selected_features = feature_cols[selected_feature_mask]
    print("Top features selected:", selected_features.tolist())


    knn_classifier = KNeighborsClassifier(k_neighbors) 
    knn_classifier.fit(X_train_selected, y_full)  

    return knn_classifier, selector

def train_classifier_aspect_ratio(df_train):

    print("Aspect Ratios and Labels for Each Object:")

    for idx, row in df_train.iterrows():
        print(f"Object {idx+1}: Aspect Ratio = {row['aspect_ratio']}, Label = {row['class']}")

    group_means = df_train.groupby("class")["aspect_ratio"].mean()
    
    # For simplicity, this classifier is intended for binary classification.
    if len(group_means) != 2:
        raise ValueError("This aspect ratio classifier expects exactly two classes.")
    
    # Sort the groups by their mean aspect ratio so we know which is lower and which is higher.
    sorted_groups = group_means.sort_values()
    lower_class = sorted_groups.index[0]
    higher_class = sorted_groups.index[1]
    cutoff = (sorted_groups.iloc[0] + sorted_groups.iloc[1]) / 2.0
    
    print("Aspect Ratio Cutoff:", cutoff)


    return {"cutoff": cutoff, "lower_class": lower_class, "higher_class": higher_class}


### -------------------------------------------------------------------------------- ###
### --------------------------- Classifiers for prediction  ------------------------ ###
### -------------------------------------------------------------------------------- ###

def classify_sample(x, class_params, method="bayes_big"):

    if method == "bayes_big":
        return classify_sample_bayes_big(x, class_params)
    elif method == "bayes_naive":
        return classify_sample_bayes_naive(x, class_params)
    elif method == "kNN":
        return classify_sample_kNN(x, class_params)
    else:
        print(f"---!! Unsupported classifier method {method}. Using 'bayes_big'.")
        return classify_sample_bayes_big(x, class_params)


def classify_sample_bayes_big(x, class_params):

    probabilities = {}

    for cls, (mean_vec, cov_mat, prior) in class_params.items():
        cov_mat_adjusted = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
        likelihood = multivariate_normal.pdf(x, mean=mean_vec, cov=cov_mat_adjusted, allow_singular=True)
        posterior = likelihood * prior
        probabilities[cls] = posterior
    predicted_class = max(probabilities, key=probabilities.get)

    return predicted_class, probabilities

def classify_sample_bayes_naive(x_value, class_params):
    
    probabilities = {}
    for cls, (mean_vec, var_vec, prior) in class_params.items():
        likelihoods = (1.0 / np.sqrt(2 * np.pi * var_vec)) * np.exp(- ((x_value - mean_vec) ** 2) / (2 * var_vec))
        likelihood = np.prod(likelihoods)
        posterior = likelihood * prior
        probabilities[cls] = posterior

    predicted_class = max(probabilities, key=probabilities.get)
    
    return predicted_class, probabilities


def classify_sample_aspect_ratio(x, aspect_params):
    aspect_value = x["aspect_ratio"]

    # Compare the value with the cutoff:
    if aspect_value <= aspect_params["cutoff"]:
        predicted_class = aspect_params["lower_class"]
    else:
        predicted_class = aspect_params["higher_class"]
    
    # Create a dummy probability dictionary with 100% probability for the predicted class.
    dummy_probability = {predicted_class: 1.0}

    return predicted_class, dummy_probability

def classify_sample_kNN(x, knn_params):

    knn_classifier, selector = knn_params

    x = x.reshape(1, -1)
    x_selected = selector.transform(x)
    predicted_class = knn_classifier.predict(x_selected)[0]
    proba = knn_classifier.predict_proba(x_selected)[0]

    probabilities = {cls: p for cls, p in zip(knn_classifier.classes_, proba)}

    return predicted_class, probabilities
    