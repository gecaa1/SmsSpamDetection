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
