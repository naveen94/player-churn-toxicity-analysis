import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

DATA = "data/chat_data.csv"
MODEL = "models/toxicity_model.pkl"

df = pd.read_csv(DATA)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["toxic"], test_size=0.2,
    random_state=42, stratify=df["toxic"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                              ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL)
print("Saved:", MODEL)
