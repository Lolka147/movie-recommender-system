import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка и подготовка данных
movies = pd.read_csv('movies.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(movie_title):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices]
    except IndexError:
        return None


# Основной цикл программы
while True:
    print("\n" + "=" * 50)
    movie_title = input("Введите название фильма (или 'quit' для выхода): ")

    if movie_title.lower() == 'quit':
        break

    recommendations = get_recommendations(movie_title)

    if recommendations is None:
        print("Фильм не найден! Проверьте название.")
    else:
        print(f"\nРекомендации для '{movie_title}':")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")