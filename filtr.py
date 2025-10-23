import pandas as pd

movies = pd.read_csv('movies.csv')

# Фильтруем комедии
horror_movies = movies[movies['genres'].str.contains('Horror')]

print(f"Найдено {len(horror_movies)} ужасов:")
print(horror_movies[['title', 'genres']].head(10))

# Теперь найди 5 случайных драм
print("\n" + "="*50)
drama_movies = movies[movies['genres'].str.contains('Drama')]
print(f"Найдено {len(drama_movies)} драм:")
print(drama_movies[['title', 'genres']].sample(5))  # .sample() берет случайные строки