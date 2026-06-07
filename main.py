import re
import string
import warnings
import os
import joblib

from IPython.display import display

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

file_path = "data/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)
df = df.iloc[:, :2]
df.columns = ["label", "message"]

df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

df = df.dropna(subset=["label", "message", "label_num"])
df["label_num"] = df["label_num"].astype(int)

print(df.shape)
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

display(df.head())

df["message_length"] = df["message"].apply(len)
class_counts = df["label"].value_counts().sort_values(ascending=True)

plt.figure(figsize=(7.5, 4.8))
plt.hlines(
    y=class_counts.index,
    xmin=0,
    xmax=class_counts.values,
    linewidth=3
)
plt.scatter(
    class_counts.values,
    class_counts.index,
    s=160,
    zorder=3
)

for label, value in class_counts.items():
    percentage = value / class_counts.sum() * 100
    plt.text(
        value + 60,
        label,
        f"{value} ({percentage:.1f}%)",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Rozkład klas w zbiorze SMS", fontsize=17, fontweight="bold", pad=14)
plt.xlabel("Liczba wiadomości")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.xlim(0, class_counts.max() * 1.18)
plt.tight_layout()
plt.savefig("results/plots/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
length_by_class = df.groupby("label")["message_length"].mean().sort_values(ascending=True)

plt.figure(figsize=(7.5, 4.8))
plt.hlines(
    y=length_by_class.index,
    xmin=0,
    xmax=length_by_class.values,
    linewidth=3
)
plt.scatter(
    length_by_class.values,
    length_by_class.index,
    s=160,
    zorder=3
)

for label, value in length_by_class.items():
    plt.text(
        value + 2,
        label,
        f"{value:.1f} znaków",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Średnia długość wiadomości według klasy", fontsize=17, fontweight="bold", pad=14)
plt.xlabel("Średnia liczba znaków")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.xlim(0, length_by_class.max() * 1.2)
plt.tight_layout()
plt.savefig("results/plots/message_length.png", dpi=300, bbox_inches="tight")
plt.show()

df.describe(include="all").to_csv("results/tables/dataset_description.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_message"] = df["message"].apply(clean_text)

display(df[["message", "clean_message"]].head())
def extract_features(text):
    text = str(text)

    return pd.Series({
        "length": len(text),
        "num_digits": sum(c.isdigit() for c in text),
        "num_exclamation": text.count("!"),
        "num_uppercase": sum(c.isupper() for c in text),
        "has_url": int(bool(re.search(r"http\S+|www\S+", text))),
        "num_words": len(text.split())
    })


extra_features = df["message"].apply(extract_features)

display(extra_features.head())

extra_features.to_csv("results/tables/extra_features.csv", index=False)

X_text = df["clean_message"]
X_extra = extra_features
y = df["label_num"]

X_train_text, X_test_text, X_train_extra, X_test_extra, y_train, y_test = train_test_split(
    X_text,
    X_extra,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

scaler = StandardScaler()
X_train_extra_scaled = scaler.fit_transform(X_train_extra)
X_test_extra_scaled = scaler.transform(X_test_extra)

X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_extra_scaled)])
X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_extra_scaled)])

print(X_train_final.shape)
print(X_test_final.shape)

joblib.dump(tfidf, "results/models/tfidf_vectorizer.pkl")
joblib.dump(scaler, "results/models/scaler.pkl")

results = []

def safe_filename(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    results.append(result)

    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(report)

    filename = safe_filename(name)

    with open(f"results/reports/classification_report_{filename}.txt", "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["actual_ham", "actual_spam"],
        columns=["predicted_ham", "predicted_spam"]
    )

    cm_df.to_csv(f"results/tables/confusion_matrix_{filename}.csv")

    cm_percent = cm / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_title(
        f"Macierz pomyłek — {name}",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Klasa przewidziana", fontsize=12)
    ax.set_ylabel("Klasa rzeczywista", fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ham", "spam"], fontsize=12)
    ax.set_yticklabels(["ham", "spam"], fontsize=12)

    for i in range(2):
        for j in range(2):
            text = (
                f"{cm[i, j]} przypadków\n"
                f"{cm_percent[i, j]:.1f}% zbioru testowego"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Liczba przypadków", fontsize=11)

    plt.tight_layout()

    plt.savefig(
        f"results/plots/confusion_matrix_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    joblib.dump(model, f"results/models/model_{filename}.pkl")

    return model
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

trained_models = {}

for name, model in models.items():

    if name == "Multinomial Naive Bayes":
        trained_models[name] = evaluate_model(
            name,
            model,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test
        )
    else:
        trained_models[name] = evaluate_model(
            name,
            model,
            X_train_final,
            X_test_final,
            y_train,
            y_test
        )
