import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')

# Подготовка данных
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(movie_title):
    # Находим индекс фильма
    idx = movies[movies['title'] == movie_title].index[0]

    # Получаем попарные схожести со всеми фильмами
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортируем по убыванию схожести
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Берем топ-5 похожих (исключаем сам фильм)
    sim_scores = sim_scores[1:6]

    # Получаем индексы похожих фильмов
    movie_indices = [i[0] for i in sim_scores]

    # Возвращаем названия
    return movies['title'].iloc[movie_indices]


# Тестируем
print("Рекомендации для 'Toy Story (1995)':")
print(get_recommendations('Toy Story (1995)'))