SMS Spam Detection
Projekt dotyczący klasyfikacji wiadomości typu spam z wykorzystaniem metod przetwarzania naturalnego (NLP) oraz uczenia maszynowego.

Celem projektu było stworzenie kompletnego pipeline'u analizy tekstu umożliwiającego wykrywanie wiadomości spamowych na podstawie treści SMS-ów.

W projekcie zaimplementowano:
-preprocessing tekstu,
-ekstrakcję cech tekstowych,
-reprezentację TF-IDF,
-porównanie różnych modeli klasyfikacji,
-tuning hiperparametrów,
-analizę ROC/AUC,
-analizę macierzy pomyłek,
-analizę błędów klasyfikacji.

Wykorzystane modele:
-Multinominal Naive Bayes
-Logistic Regression
-Linear SVM
-Random Forest
-Gradient Boosting
-Voting Classifier

Projekt wykorzystuje zbiór:
SMS Spam Collection Dataset
Parametry zbioru:
* 5572 wiadomości SMS,
* klasy:
    * ham — wiadomości poprawne,
    * spam — wiadomości reklamowe / niepożądane.


Wymagania:
* Python 3.13
* pip
* venv

Instalacja i uruchomienie
Utworzenie środowiska:
python -m venv .venv
Aktywacja środowiska:
source .venv/bin/activate
Instalacja zależności:
pip install -r requirements.txt
Uruchomienie projektu:
python main.py
Struktura projektu
data/       -> zbiór danych
main.py    -> kod źródłowy
results/    -> wygenerowane wykresy, raporty i modele