Opis projektu

Problem:
Spam SMS stanowi istotny problem związany z bezpieczeństwem użytkowników telefonów komórkowych. Wiadomości spamowe często zawierają treści reklamowe, próby phishingu lub wyłudzania danych.
Ze względu na dużą liczbę wiadomości konieczne jest zastosowanie automatycznych metod klasyfikacji tekstu umożliwiających wykrywanie spamu.

Wykorzystany artykuł naukowy:
https://www.sciencedirect.com/science/article/pii/S187705092302094X/pdf?md5=5008992964aa82c948924107bd3a5869&pid=1-s2.0-S187705092302094X-main.pdf
Link do dataset:
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Dotychczasowe rozwiązania:
W klasyfikacji wiadomości tekstowych często stosuje się:
* metody probabilistyczne,
* modele liniowe,
* klasyfikatory SVM,
* reprezentacje TF-IDF,
* analizę cech tekstowych.

Popularnym podejściem jest przekształcenie tekstu do reprezentacji numerycznej, a następnie wykorzystanie modeli uczenia maszynowego do klasyfikacji wiadomości.


W projekcie wykorzystano:
* preprocessing tekstu,
* usuwanie znaków specjalnych,
* reprezentację TF-IDF,
* ekstrakcję dodatkowych cech tekstowych,
* porównanie wielu modeli klasyfikacyjnych.

Porównano skuteczność następujących modeli:
* Multinomial Naive Bayes,
* Logistic Regression,
* Linear SVM,
* Random Forest,
* Gradient Boosting,
* Voting Classifier.

Dodatkowo wykonano:
* tuning hiperparametrów,
* analizę ROC/AUC,
* analizę błędów klasyfikacji,
* analizę najważniejszych cech wskazujących spam.

Wyniki
Najlepsze wyniki osiągnął model Linear SVM po dostrojeniu hiperparametrów.
Model osiągnął skuteczność klasyfikacji na poziomie około 99% accuracy oraz około 96% F1-score.