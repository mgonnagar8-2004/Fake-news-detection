import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 📥 Download stopwords
# -------------------------------
nltk.download('stopwords')

# -------------------------------
# 📊 Load Dataset
# -------------------------------
try:
    df = pd.read_csv("Fake And Real News Prediction/fake_or_real_news.csv")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

print("✅ Dataset loaded successfully!")
print("Columns:", df.columns)

# -------------------------------
# 🧹 Remove unwanted column
# -------------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# -------------------------------
# 🧠 Combine title + text
# -------------------------------
df["text"] = df["title"] + " " + df["text"]

# -------------------------------
# 🔢 Convert label (FAKE/REAL → 0/1)
# -------------------------------
df["label"] = df["label"].map({'FAKE': 0, 'REAL': 1})

# -------------------------------
# 🧹 Text Cleaning
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

print("✅ Text preprocessing completed!")

# -------------------------------
# 📌 Features & Labels
# -------------------------------
X = df["text"]
y = df["label"]

# -------------------------------
# 🔀 Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 📊 TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("✅ TF-IDF vectorization done!")

# -------------------------------
# 🤖 Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("✅ Model training completed!")

# -------------------------------
# 📈 Accuracy
# -------------------------------
accuracy = model.score(X_test_vec, y_test)
print(f"🎯 Accuracy: {round(accuracy*100, 2)}%")

# -------------------------------
# 💾 Save Model & Vectorizer
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("💾 model.pkl and vectorizer.pkl created successfully!")