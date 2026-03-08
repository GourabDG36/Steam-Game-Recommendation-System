import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\my projects\AI and MLops\steam-game-recommender\data\steam.csv\steam.csv")

print("Original dataset shape:", df.shape)

# Select useful columns
df = df[['name','genres','categories','steamspy_tags']]

# Rename tags column
df.rename(columns={'steamspy_tags':'tags'}, inplace=True)

# Remove missing values
df = df.dropna()

print("\nDataset shape after cleaning:", df.shape)

# Combine features
df['combined_features'] = df['genres'] + " " + df['categories'] + " " + df['tags']

# Keep final columns
df = df[['name','combined_features']]

print("\nSample cleaned dataset:")
print(df.head())

# Save cleaned dataset
df.to_csv("D:/my projects/AI and MLops/steam-game-recommender/data/cleaned_games.csv", index=False)

print("\nCleaned dataset saved successfully.")