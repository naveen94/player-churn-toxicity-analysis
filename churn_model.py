import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA = "data/player_data.csv"
MODEL = "models/churn_model.pkl"

df = pd.read_csv(DATA)
features = [
    "age", "days_since_signup", "sessions_per_week",
    "avg_session_minutes", "weekly_hours", "days_inactive",
    "matches_played", "win_rate", "purchases",
    "avg_purchase_value", "support_tickets",
    "toxicity_rate", "friends_count"
]
X = df[features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))

joblib.dump({"model": model, "features": features}, MODEL)
print("Saved:", MODEL)
