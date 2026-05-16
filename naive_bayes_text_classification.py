naive_bayes_text_classification.py

import warnings
warnings.filterwarnings("ignore")  # silence minor sklearn warnings

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# 1. Load text data
# --------------------------------------------------
categories = ["rec.sport.baseball","sci.space"]  # two categories
data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers","footers","quotes") # strip metadata
)

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
print(X_train[:5])
print()
print(y_train[:5])
# --------------------------------------------------
# 2. Build a training pipeline
# --------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("nb", MultinomialNB())
])

# --------------------------------------------------
# 3. Train
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------
y_pred = model.predict(X_test)
print("\n=== Evaluation on test set ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=categories))

# --------------------------------------------------
# 5. Classify new text
# --------------------------------------------------
custom_docs = [
    "NASA will deploy a new satellite to study exoplanet atmospheres.",
    "The pitcher threw a no-hitter in last night's game."
]
pred = model.predict(custom_docs)
print("=== Predictions on custom sentences ===")
for doc, label in zip(custom_docs, pred):
    print(f"• “{doc}” ⇒ {categories[label]}")

probs = model.predict_proba(custom_docs)
for doc, prob in zip(custom_docs, probs):
    print(f"\nDocument: {doc}")
    for cat, p in zip(categories, prob):
        print(f"Probability {cat}: {p:.4f}")
