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