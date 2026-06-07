Przykład działania projektu

Przykładowe wiadomości::

Wiadomość

Predykcja

* Congratulations! You won a free prize. Click here now! | SPAM
* Hey, are we meeting tomorrow at university? | HAM
* FREE entry into our weekly competition. Reply WIN to participate. | SPAM
* Can you send me the project files when you finish? | HAM
* Your bank account requires immediate verification. Click the link below. | SPAM

Model poprawnie rozpoznaje wiadomości zawierające:
* słowa związane z nagrodami,
* podejrzane linki,
* komunikaty o wygranych,
* treści marketingowe.
Wiadomości prywatne i neutralne klasyfikowane są jako HAM.

Wyniki modeli:

* Model --------------------------- F1-score
* Linear SVM -----------------------0.95
* Random Forest ----------------------- 0.94
* Logistic Regression ----------------------- 0.93
* Naive Bayes ----------------------- 0.91

Najwyższą skuteczność osiągnął model Linear SVM wykorzystujący reprezentację TF-IDF oraz dodatkowe cechy tekstowe