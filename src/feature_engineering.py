import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "cleaned_games.csv")

df = pd.read_csv(data_path)

print("Cleaned dataset shape:", df.shape)

# Initialize TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Convert text to vectors
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Save TF-IDF matrix
with open(r"D:\my projects\AI and MLops\steam-game-recommender\models\tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

# Save vectorizer
with open(r"D:\my projects\AI and MLops\steam-game-recommender\models\vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save dataset used for recommendation
df.to_csv(r"D:\my projects\AI and MLops\steam-game-recommender\data\final_games_dataset.csv", index=False)

print("\nFeature engineering completed successfully.")