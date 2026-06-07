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
from custom_naive_bayes import CustomMultinomialNB

os.makedirs("../results", exist_ok=True)
os.makedirs("../results/plots", exist_ok=True)
os.makedirs("../results/tables", exist_ok=True)
os.makedirs("../results/models", exist_ok=True)
os.makedirs("../results/reports", exist_ok=True)

file_path = "../data/SMSSpamCollection"

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

joblib.dump(tfidf, "../results/models/tfidf_vectorizer.pkl")
joblib.dump(scaler, "../results/models/scaler.pkl")

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
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False)

display(results_df)

results_df.to_csv("results/tables/model_results.csv", index=False)
# =========================
# CUSTOM MULTINOMIAL NAIVE BAYES
# =========================

custom_nb = CustomMultinomialNB()

custom_nb.fit(
    X_train_text.tolist(),
    y_train.tolist()
)

custom_predictions = custom_nb.predict(
    X_test_text.tolist()
)

custom_result = {
    "model": "Custom Multinomial NB",
    "accuracy": accuracy_score(y_test, custom_predictions),
    "precision": precision_score(y_test, custom_predictions),
    "recall": recall_score(y_test, custom_predictions),
    "f1": f1_score(y_test, custom_predictions)
}

results.append(custom_result)

print("=" * 60)
print("Custom Multinomial Naive Bayes")
print("=" * 60)

print(classification_report(
    y_test,
    custom_predictions,
    target_names=["ham", "spam"]
))

plot_df = results_df.sort_values(by="f1", ascending=True)

plt.figure(figsize=(10, 6))
plt.hlines(
    y=plot_df["model"],
    xmin=0,
    xmax=plot_df["f1"],
    linewidth=3
)
plt.scatter(
    plot_df["f1"],
    plot_df["model"],
    s=140,
    zorder=3
)

for f1, model_name in zip(plot_df["f1"], plot_df["model"]):
    plt.text(
        f1 + 0.01,
        model_name,
        f"{f1:.3f}",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Porównanie modeli według F1-score", fontsize=17, fontweight="bold", pad=15)
plt.xlabel("F1-score")
plt.ylabel("")
plt.xlim(0, 1.05)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("results/plots/model_comparison_f1.png", dpi=300, bbox_inches="tight")
plt.show()

svm_params = {
    "C": [0.1, 1, 5, 10]
}

grid_svm = GridSearchCV(
    LinearSVC(),
    svm_params,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_svm.fit(X_train_final, y_train)

print("Najlepsze parametry:", grid_svm.best_params_)
print("Najlepszy wynik CV:", grid_svm.best_score_)

grid_results_df = pd.DataFrame(grid_svm.cv_results_)
grid_results_df.to_csv("results/tables/grid_search_svm_results.csv", index=False)

with open("../results/reports/grid_search_svm_best_params.txt", "w", encoding="utf-8") as file:
    file.write(f"Najlepsze parametry: {grid_svm.best_params_}\n")
    file.write(f"Najlepszy wynik CV: {grid_svm.best_score_}\n")

best_svm = evaluate_model(
    "Tuned Linear SVM",
    grid_svm.best_estimator_,
    X_train_final,
    X_test_final,
    y_train,
    y_test
)

voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("svc", SVC(probability=True, kernel="linear", C=1)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    voting="soft"
)

voting = evaluate_model(
    "Voting Classifier",
    voting,
    X_train_final,
    X_test_final,
    y_train,
    y_test
)
roc_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Voting Classifier": voting
}

roc_results = []

plt.figure(figsize=(8.5, 6.2))

for name, model in roc_models.items():
    model.fit(X_train_final, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_final)[:, 1]
    else:
        continue

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    roc_results.append({
        "model": name,
        "auc": roc_auc
    })

    plt.plot(
        fpr,
        tpr,
        linewidth=2.4,
        label=f"{name} | AUC = {roc_auc:.3f}"
    )

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Losowy klasyfikator")
plt.title("Krzywa ROC dla wybranych modeli", fontsize=17, fontweight="bold", pad=15)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(linestyle="--", alpha=0.35)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("results/plots/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

roc_results_df = pd.DataFrame(roc_results)
roc_results_df.to_csv("results/tables/roc_auc_results.csv", index=False)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

feature_names = np.array(tfidf.get_feature_names_out())
coefficients = lr.coef_[0]

top_spam_idx = coefficients.argsort()[-20:]
top_ham_idx = coefficients.argsort()[:20]

top_spam_words = pd.DataFrame({
    "word": feature_names[top_spam_idx],
    "coefficient": coefficients[top_spam_idx]
}).sort_values(by="coefficient", ascending=False)

top_ham_words = pd.DataFrame({
    "word": feature_names[top_ham_idx],
    "coefficient": coefficients[top_ham_idx]
}).sort_values(by="coefficient")

print("Najważniejsze słowa dla SPAM:")
display(top_spam_words)

print("Najważniejsze słowa dla HAM:")
display(top_ham_words)

top_spam_words.to_csv("results/tables/top_spam_words.csv", index=False)
top_ham_words.to_csv("results/tables/top_ham_words.csv", index=False)

plot_words = top_spam_words.sort_values(by="coefficient", ascending=True)

plt.figure(figsize=(9.5, 7.2))
plt.hlines(
    y=plot_words["word"],
    xmin=0,
    xmax=plot_words["coefficient"],
    linewidth=3
)
plt.scatter(
    plot_words["coefficient"],
    plot_words["word"],
    s=110,
    zorder=3
)

for coef, word in zip(plot_words["coefficient"], plot_words["word"]):
    plt.text(
        coef + 0.08,
        word,
        f"{coef:.2f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Najsilniejsze cechy wskazujące na SPAM", fontsize=17, fontweight="bold", pad=15)
plt.xlabel("Waga cechy w modelu Logistic Regression")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.xlim(0, plot_words["coefficient"].max() * 1.18)
plt.tight_layout()
plt.savefig("results/plots/top_spam_words.png", dpi=300, bbox_inches="tight")
plt.show()

best_model = grid_svm.best_estimator_
best_model.fit(X_train_final, y_train)

y_pred = best_model.predict(X_test_final)

errors = pd.DataFrame({
    "message": X_test_text.values,
    "true_label": y_test.values,
    "predicted_label": y_pred
})

errors = errors[errors["true_label"] != errors["predicted_label"]]

errors["true_label"] = errors["true_label"].map({0: "ham", 1: "spam"})
errors["predicted_label"] = errors["predicted_label"].map({0: "ham", 1: "spam"})

display(errors.head(20))

errors.to_csv("results/tables/model_errors.csv", index=False)

count_vectorizer = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

comparison_results = []

for vectorizer_name, Xtr, Xte in [
    ("CountVectorizer", X_train_count, X_test_count),
    ("TF-IDF", X_train_tfidf, X_test_tfidf)
]:
    model = LinearSVC(C=1)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    comparison_results.append({
        "representation": vectorizer_name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred)
    })

comparison_df = pd.DataFrame(comparison_results)

display(comparison_df)

comparison_df.to_csv("results/tables/vectorizer_comparison.csv", index=False)

plot_df = comparison_df.sort_values(by="f1", ascending=True)

plt.figure(figsize=(8.5, 4.8))
plt.hlines(
    y=plot_df["representation"],
    xmin=0,
    xmax=plot_df["f1"],
    linewidth=3
)
plt.scatter(
    plot_df["f1"],
    plot_df["representation"],
    s=160,
    zorder=3
)

for f1, representation in zip(plot_df["f1"], plot_df["representation"]):
    plt.text(
        f1 + 0.01,
        representation,
        f"{f1:.3f}",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Porównanie reprezentacji tekstu", fontsize=17, fontweight="bold", pad=15)
plt.xlabel("F1-score")
plt.ylabel("")
plt.xlim(0, 1.05)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("results/plots/vectorizer_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

def predict_sms(text):

    cleaned = clean_text(text)

    text_tfidf = tfidf.transform([cleaned])

    text_extra = extract_features(text).to_frame().T

    text_extra_scaled = scaler.transform(text_extra)

    final_features = hstack([text_tfidf, csr_matrix(text_extra_scaled)])

    prediction = best_model.predict(final_features)[0]

    return "SPAM" if prediction == 1 else "HAM"

custom_messages_path = "../data/custom_messages.txt"

with open(custom_messages_path, "r", encoding="utf-8") as file:

    examples = [

        line.strip()

        for line in file.readlines()

        if line.strip()

    ]

prediction_results = []

for sms in examples:

    prediction = predict_sms(sms)

    prediction_results.append({

        "message": sms,

        "prediction": prediction

    })

    print(sms, "->", prediction)

prediction_df = pd.DataFrame(prediction_results)

prediction_df.to_csv("results/tables/sample_predictions.csv", index=False)

display(prediction_df)

joblib.dump(best_model, "../results/models/best_model_tuned_linear_svm.pkl")

print("ukonczono")