import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(r"D:\my projects\AI and MLops\steam-game-recommender\data\final_games_dataset.csv")

# Load TF-IDF matrix
with open(r"D:\my projects\AI and MLops\steam-game-recommender\models\tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity matrix shape:", similarity_matrix.shape)


def recommend(game_name, top_n=5):

    # Check if game exists
    if game_name not in df['name'].values:
        print("Game not found in dataset")
        return

    # Get index of game
    game_index = df[df['name'] == game_name].index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(similarity_matrix[game_index]))

    # Sort by similarity
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_games = similarity_scores[1:top_n+1]

    print(f"\nGames similar to '{game_name}':\n")

    for i in top_games:
        print(df.iloc[i[0]]['name'])


# Test recommendation
recommend("Counter-Strike")