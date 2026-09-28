# Player Churn Analysis & Toxicity Detection

A Data & Business Analytics project that combines **player retention analysis**, **churn prediction**, and **NLP-based toxicity detection** for an online gaming environment.

## Project Objective

Gaming businesses need to answer two important questions:

1. Which players are at risk of leaving the platform?
2. Which player messages may negatively affect community health?

This project uses player activity data and chat-message data to produce analytical insights and predictive models.

## Key Features

### Player Churn Analysis
- Data cleaning and exploratory analysis
- Player engagement analysis
- Churn-rate measurement
- Churn-driver analysis
- Random Forest churn prediction
- Individual churn-risk prediction

### Toxicity Detection
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classifier
- Toxic/non-toxic classification
- Confidence score for sample messages

### Business Analytics
- Retention KPIs
- Community-health KPI
- Player segmentation
- Action-oriented business insights
- Interactive Streamlit dashboard

## Dataset

This repository includes **synthetic data generated for demonstration and educational purposes**. It is not real customer or player data.

Files:
- `data/player_data.csv` — 1,500 synthetic player records
- `data/chat_data.csv` — 3,000 synthetic chat messages

## Project Structure

```text
player-churn-toxicity-analysis/
├── app/
│   └── app.py
├── data/
│   ├── player_data.csv
│   └── chat_data.csv
├── models/
├── notebooks/
│   ├── churn_analysis.py
│   └── toxicity_analysis.py
├── src/
│   ├── churn_model.py
│   └── toxicity_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/player-churn-toxicity-analysis.git
cd player-churn-toxicity-analysis

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Models

From the project root:

```bash
python src/churn_model.py
python src/toxicity_model.py
```

The trained models will be saved inside:

```text
models/
```

## Run the Dashboard

```bash
streamlit run app/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Business Questions Answered

- What percentage of players churn?
- How does inactivity relate to churn?
- How does engagement relate to retention?
- Which player characteristics are associated with churn?
- What percentage of chat messages are toxic?
- How can toxicity monitoring support community management?
- How can a gaming company use churn risk for targeted retention programs?

## Important Note

The churn labels and toxicity labels in this demonstration dataset are synthetic. Model performance on this dataset should **not** be interpreted as real-world business performance.

For a production system, the models should be validated using real historical player outcomes and professionally labeled moderation data.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Random Forest
- Matplotlib
- Seaborn
- Streamlit
- GitHub

## Author

**Naveen K**

Computer Science and Business Systems  
Ramco Institute of Technology
