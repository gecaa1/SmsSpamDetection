# Opis projektu

## Problem

Spam SMS stanowi istotny problem związany z bezpieczeństwem użytkowników telefonów komórkowych. Wiadomości spamowe często zawierają treści reklamowe, próby phishingu lub wyłudzania danych. Ze względu na dużą liczbę wiadomości konieczne jest zastosowanie automatycznych metod klasyfikacji tekstu umożliwiających wykrywanie spamu.

Celem projektu było przygotowanie systemu klasyfikacji wiadomości SMS z wykorzystaniem metod uczenia maszynowego oraz porównanie skuteczności różnych modeli klasyfikacyjnych.

---

## Tytuł artykułu bazowego

SMS Spam Detection using Relevance Vector Machine

## Wykorzystany artykuł naukowy

https://www.sciencedirect.com/science/article/pii/S187705092302094X/pdf

---

## Dataset

W projekcie wykorzystano publiczny zbiór danych SMS Spam Collection Dataset:

https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Zbiór zawiera wiadomości oznaczone jako:
- spam,
- ham (wiadomości poprawne).

---

## Podstawy teoretyczne

Projekt dotyczy problemu klasyfikacji binarnej, w której każda wiadomość SMS przypisywana jest do jednej z dwóch klas:
- spam,
- non-spam.

W klasyfikacji tekstu wiadomości muszą zostać przekształcone do postaci numerycznej. W projekcie zastosowano reprezentację TF-IDF (Term Frequency – Inverse Document Frequency), która przypisuje większą wagę słowom istotnym dla danej wiadomości.

TF-IDF można zapisać jako:

TFIDF(t,d) = TF(t,d) * IDF(t)

gdzie:

IDF(t) = log(N / df(t))

- t — słowo,
- d — dokument,
- N — liczba dokumentów,
- df(t) — liczba dokumentów zawierających dane słowo.

---

## Multinomial Naive Bayes

W projekcie wykorzystano również model Multinomial Naive Bayes. Model bazuje na twierdzeniu Bayesa:

P(C|x) = (P(x|C) * P(C)) / P(x)

gdzie:
- C oznacza klasę wiadomości,
- x oznacza analizowaną wiadomość,
- P(C|x) oznacza prawdopodobieństwo przynależności wiadomości do klasy.

W modelu Multinomial Naive Bayes wiadomość traktowana jest jako zbiór słów, a klasyfikacja wykonywana jest na podstawie częstotliwości ich występowania.

Aby uniknąć zerowych prawdopodobieństw dla nowych słów zastosowano wygładzanie Laplace’a:

P(w|C) = (count(w,C)+1) / (N_C+|V|)

gdzie:
- count(w,C) — liczba wystąpień słowa w klasie,
- N_C — liczba wszystkich słów w klasie,
- |V| — liczba słów w słowniku.

---

## Dotychczasowe rozwiązania

W klasyfikacji wiadomości tekstowych często stosuje się:
- metody probabilistyczne,
- modele liniowe,
- klasyfikatory SVM,
- reprezentacje TF-IDF,
- analizę cech tekstowych.

Popularnym podejściem jest przekształcenie tekstu do reprezentacji numerycznej, a następnie wykorzystanie modeli uczenia maszynowego do klasyfikacji wiadomości.

---

## Zastosowane podejście

W projekcie wykorzystano:
- preprocessing tekstu,
- usuwanie znaków specjalnych,
- konwersję tekstu do małych liter,
- usuwanie znaków interpunkcyjnych,
- reprezentację TF-IDF,
- ekstrakcję dodatkowych cech tekstowych,
- porównanie wielu modeli klasyfikacyjnych.

Dodatkowo zastosowano feature engineering obejmujący:
- długość wiadomości,
- liczbę cyfr,
- liczbę wykrzykników,
- liczbę wielkich liter,
- obecność linków,
- liczbę słów.

---

## Wykorzystane modele

Porównano skuteczność następujących modeli:
- Multinomial Naive Bayes,
- Custom Multinomial Naive Bayes,
- Logistic Regression,
- Linear SVM,
- Random Forest,
- Gradient Boosting,
- Voting Classifier.

Dodatkowo wykonano:
- tuning hiperparametrów,
- analizę ROC/AUC,
- analizę błędów klasyfikacji,
- analizę najważniejszych cech wskazujących spam.

---

## Istotne elementy implementacji

W projekcie część algorytmów została zaimplementowana samodzielnie.

Przygotowano własną implementację modelu Custom Multinomial Naive Bayes, która:
- buduje słownik słów na podstawie danych treningowych,
- zlicza wystąpienia słów dla klas spam i ham,
- oblicza prawdopodobieństwa klas,
- stosuje wygładzanie Laplace’a,
- klasyfikuje nowe wiadomości na podstawie logarytmicznych prawdopodobieństw.

Biblioteka scikit-learn została wykorzystana do porównania wyników własnej implementacji z popularnymi modelami uczenia maszynowego.

---

## Nowe podejście względem artykułu

Artykuł bazowy skupiał się głównie na porównaniu metod:
- Naive Bayes,
- Support Vector Machine,
- Relevance Vector Machine.

W naszej implementacji projekt został rozszerzony o:
- własną implementację Multinomial Naive Bayes,
- dodatkowe modele klasyfikacyjne,
- feature engineering,
- porównanie CountVectorizer oraz TF-IDF,
- tuning hiperparametrów,
- analizę ROC/AUC,
- analizę błędów klasyfikacji,
- analizę najważniejszych słów wpływających na klasyfikację.

---

## Wyniki

Najlepsze wyniki osiągnął model Linear SVM po dostrojeniu hiperparametrów.

Model osiągnął:
- accuracy ≈ 0.99,
- F1-score ≈ 0.96.

Własna implementacja Custom Multinomial Naive Bayes osiągnęła:
- accuracy = 0.981,
- precision = 0.910,
- recall = 0.953,
- F1-score = 0.931.

Wyniki pokazują, że klasyczne modele uczenia maszynowego nadal bardzo dobrze sprawdzają się w zadaniach klasyfikacji wiadomości SMS.