import pandas as pd

df = pd.read_csv(r"D:\my projects\AI and MLops\steam-game-recommender\data\games.csv")

df.head()
df.shape
df.info()
df.isnull().sum()