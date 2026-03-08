import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# Load Data
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "final_games_dataset.csv")
model_path = os.path.join(BASE_DIR, "models", "tfidf_matrix.pkl")

df = pd.read_csv(data_path)

with open(model_path, "rb") as f:
    tfidf_matrix = pickle.load(f)

# -----------------------------
# Recommendation Function
# -----------------------------

def recommend(game_name, top_n=5):

    game_index = df[df['name'] == game_name].index[0]

    game_vector = tfidf_matrix[game_index]

    similarity_scores = cosine_similarity(game_vector, tfidf_matrix).flatten()

    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]

    recommended_games = df.iloc[similar_indices]['name'].values

    return recommended_games


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎮 Steam Game Recommendation System")

st.write("Search for a game and get similar recommendations.")

# Game search
search_query = st.text_input("Search for a game")

filtered_games = df[df['name'].str.contains(search_query, case=False, na=False)]

selected_game = None

if not filtered_games.empty:
    selected_game = st.selectbox("Select a game", filtered_games['name'])

# Recommend button
if st.button("Recommend") and selected_game:

    recommendations = recommend(selected_game)

    st.subheader("Recommended Games")

    for game in recommendations:

        game_row = df[df['name'] == game]
        features = game_row['combined_features'].values[0]

        st.write("###", game)
        st.caption(features[:120] + "...")