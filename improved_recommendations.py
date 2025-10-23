import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_detailed_recommendations(movie_title):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Топ-5

        print(f"\nФильм: {movie_title}")
        print(f"Жанры: {movies.iloc[idx]['genres']}")
        print("\nРекомендации:")

        for i, (movie_idx, score) in enumerate(sim_scores, 1):
            movie = movies.iloc[movie_idx]
            print(f"{i}. {movie['title']} (схожесть: {score:.3f})")
            print(f"   Жанры: {movie['genres']}\n")

    except IndexError:
        print("Фильм не найден!")


# Тестируем на нескольких фильмах
test_movies = ['Toy Story (1995)', 'Godfather, The (1972)', 'Matrix, The (1999)']
for movie in test_movies:
    get_detailed_recommendations(movie)