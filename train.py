import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/bug_reports.csv")
print("Loaded rows:", len(df))

df["summary"] = df["summary"].astype(str).replace("nan", "")
df["description"] = df["description"].astype(str).replace("nan", "")
df["text"] = df["summary"] + " " + df["description"]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.3, random_state=42
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, zero_division=0))

joblib.dump(pipe, "models/bug_classifier_pipeline.pkl")
print("Model saved to models/bug_classifier_pipeline.pkl")