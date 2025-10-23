import pandas as pd

movies = pd.read_csv('movies.csv')


def find_similar_movies(movie_title):
    # Находим фильм который ввел пользователь
    movie = movies[movies['title'] == movie_title]

    if len(movie) == 0:
        print("Фильм не найден! Проверь название.")
        return

    # Получаем жанры этого фильма
    movie_genres = movie['genres'].iloc[0]
    print(f"Жанры фильма '{movie_title}': {movie_genres}")

    # Ищем фильмы с хотя бы одним совпадающим жанром
    similar_movies = movies[movies['genres'].str.contains('|'.join(movie_genres.split('|')))]

    # Исключаем исходный фильм
    similar_movies = similar_movies[similar_movies['title'] != movie_title]

    print(f"\nНайдено {len(similar_movies)} похожих фильмов:")
    print(similar_movies[['title', 'genres']].head(5))


# Тестируем
find_similar_movies("Toy Story (1995)")