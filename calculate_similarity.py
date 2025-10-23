import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')

# Создаем матрицу схожести
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Матрица схожести создана!")
print("Размер:", cosine_sim.shape)

# Проверяем схожесть первых двух фильмов
print(f"\nСхожесть между фильмом 0 и 1: {cosine_sim[0][1]:.3f}")