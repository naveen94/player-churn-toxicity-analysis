# Player Churn Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/player_data.csv")
print(df.head())
print(df.describe())

print("\nChurn rate:", df["churned"].mean())

# Business comparison
print(df.groupby("churned")[[
    "sessions_per_week", "avg_session_minutes", "days_inactive",
    "win_rate", "purchases", "toxicity_rate", "friends_count"
]].mean())

# Example visualizations
sns.countplot(data=df, x="churned")
plt.title("Player Churn Distribution")
plt.show()

sns.boxplot(data=df, x="churned", y="days_inactive")
plt.title("Inactivity vs Churn")
plt.show()
