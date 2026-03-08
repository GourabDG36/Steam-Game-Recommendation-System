# 🎮 Steam Game Recommendation System

A **Machine Learning-based content recommendation system** that suggests similar Steam games based on their genres, categories, and tags.

The system uses **TF-IDF vectorization** and **cosine similarity** to identify games with similar characteristics and provides recommendations through an **interactive Streamlit web application**.

---

# 🚀 Features

* 🔎 Search for any Steam game
* 🎯 Get top **5 similar game recommendations**
* 📊 Uses **27,000+ Steam games dataset**
* ⚡ Fast similarity search using **TF-IDF + cosine similarity**
* 🖥️ Interactive **Streamlit web interface**
* 🧠 Explainable recommendations based on game features

---

# 🧠 Machine Learning Pipeline

```
Steam Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering (TF-IDF)
      ↓
Cosine Similarity Model
      ↓
Recommendation Engine
      ↓
Streamlit Web Application
```

---

# 📂 Project Structure

```
steam-game-recommender
│
├── data
│   ├── steam_games.csv
│   ├── cleaned_games.csv
│   └── final_games_dataset.csv
│
├── models
│   ├── tfidf_matrix.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── recommender.py
│
├── app
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/GourabDG36/steam-game-recommender.git
cd steam-game-recommender
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app/app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

# 🖥️ Example Output

Search for a game:

```
Dota 2
```

Recommended games:

```
Line of Sight
Awesomenauts
Counter-Strike: Global Offensive
Khan: Absolute Power
Block N Load
```

---

# 🛠 Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Cosine Similarity**
* **Streamlit**

---

# 📊 Dataset

Steam game dataset containing:

* Game names
* Genres
* Categories
* Tags

Total records: **27,000+ games**

---

# 📈 Future Improvements

* Add **Steam game cover images**
* Use **FAISS for faster similarity search**
* Deploy the app online
* Add **user-based recommendations**
* Include **game ratings and popularity**

---

# 👨‍💻 Author

**Gourabdipta Ghosh**

Machine Learning & AI Enthusiast

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!

