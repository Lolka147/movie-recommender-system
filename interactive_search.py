import pandas as pd

movies = pd.read_csv('movies.csv')


def find_movies_by_genre():
    genre = input("Введите жанр для поиска (на английском, например: Comedy, Drama, Horror): ")

    # Фильтруем по введенному жанру
    filtered_movies = movies[movies['genres'].str.contains(genre)]

    if len(filtered_movies) == 0:
        print(f"Фильмов в жанре '{genre}' не найдено.")
    else:
        print(f"\nНайдено {len(filtered_movies)} фильмов в жанре '{genre}':")
        print(filtered_movies[['title', 'genres']].head(10))


# Запускаем функцию
find_movies_by_genre()