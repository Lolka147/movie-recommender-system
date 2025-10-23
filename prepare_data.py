import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Загружаем данные
movies = pd.read_csv('movies.csv')

# Создаем объект для преобразования текста в числа
tfidf = TfidfVectorizer(stop_words='english')

# Преобразуем жанры в числовые векторы
tfidf_matrix = tfidf.fit_transform(movies['genres'])

print("Размерность матрицы признаков:", tfidf_matrix.shape)
print("\nНазвания признаков (жанры):", tfidf.get_feature_names_out())