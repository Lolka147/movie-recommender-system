import pandas as pd

# Загружаем данные
movies = pd.read_csv('movies.csv')  # Укажи правильный путь к файлу!

# Смотрим на структуру
print("--- Первые 5 строк ---")
print(movies.head())

print("\n--- Размер таблицы ---")
print(movies.shape)

print("\n--- Названия колонок ---")
print(movies.columns)

# Простой анализ: сколько фильмов каждого жанра
print("\n--- Примеры жанров ---")
print(movies['genres'].head(10))