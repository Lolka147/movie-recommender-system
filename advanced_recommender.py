import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка всех данных
@st.cache_data
def load_data():
    # Фильмы - читаем как строки
    movies = pd.read_csv('movies.csv', header=None, names=['movieId', 'title', 'genres'])
    movies['genres'] = movies['genres'].str.rstrip(';').fillna('Unknown')
    movies = movies.dropna()

    # Рейтинги (если есть файл ratings.csv)
    try:
        ratings = pd.read_csv('ratings.csv')

        # ПРЕОБРАЗУЕМ ТИПЫ ДАННЫХ!
        movies['movieId'] = movies['movieId'].astype(str)
        ratings['movieId'] = ratings['movieId'].astype(str)

        # Расчет среднего рейтинга для каждого фильма
        movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

        # Объединяем с фильмами
        movies = movies.merge(movie_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(0)
        movies['rating_count'] = movies['rating_count'].fillna(0)

    except FileNotFoundError:
        st.warning("Файл ratings.csv не найден. Работаем без рейтингов.")
        movies['avg_rating'] = 0
        movies['rating_count'] = 0

    return movies


movies = load_data()


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, tfidf


cosine_sim, tfidf = create_model()


# Улучшенная функция рекомендаций
def get_enhanced_recommendations(movie_title, weight_rating=0.3, top_n=5):
    idx = movies[movies['title'] == movie_title].index[0]

    # Базовые рекомендации по схожести
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Учитываем рейтинги в рекомендациях
    enhanced_scores = []
    for i, score in sim_scores:
        if i != idx:  # Исключаем исходный фильм
            rating_score = movies.iloc[i]['avg_rating'] / 5.0  # Нормализуем рейтинг
            combined_score = (1 - weight_rating) * score + weight_rating * rating_score
            enhanced_scores.append((i, combined_score, score, rating_score))

    # Сортируем по комбинированному score
    enhanced_scores = sorted(enhanced_scores, key=lambda x: x[1], reverse=True)

    # Берем топ-N
    enhanced_scores = enhanced_scores[:top_n]

    return enhanced_scores


# Интерфейс
st.set_page_config(page_title="Умный Рекомендатель Фильмов", layout="wide")

st.title("🎬 Умная Система Рекомендаций Фильмов")
st.markdown("---")

# Сайдбар с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    weight_rating = st.slider("Влияние рейтинга на рекомендации", 0.0, 1.0, 0.3)
    top_n = st.slider("Количество рекомендаций", 3, 10, 5)

    st.header("📊 Статистика")
    st.write(f"Фильмов в базе: **{len(movies)}**")
    if 'avg_rating' in movies.columns:
        st.write(f"Фильмов с рейтингами: **{len(movies[movies['rating_count'] > 0])}**")

# Основной контент
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Выбор фильма")
    selected_movie = st.selectbox("Выберите фильм для рекомендаций:", movies['title'].values)

    # Информация о выбранном фильме
    if selected_movie:
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        st.write(f"**Жанры:** {movie_info['genres']}")
        if movie_info['avg_rating'] > 0:
            st.write(f"**Рейтинг:** ⭐ {movie_info['avg_rating']:.1f}/5")
            st.write(f"**Оценок:** {int(movie_info['rating_count'])}")

with col2:
    if st.button("🎯 Получить Умные Рекомендации", type="primary"):
        with st.spinner("Ищем лучшие рекомендации..."):
            recommendations = get_enhanced_recommendations(selected_movie, weight_rating, top_n)

            st.subheader("💡 Рекомендуем посмотреть:")

            for i, (movie_idx, combined_score, genre_score, rating_score) in enumerate(recommendations, 1):
                movie = movies.iloc[movie_idx]

                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"**{i}. {movie['title']}**")
                        st.write(f"_{movie['genres']}_")

                    with col2:
                        if movie['avg_rating'] > 0:
                            st.write(f"⭐ {movie['avg_rating']:.1f}")
                        else:
                            st.write("⭐ --")

                    with col3:
                        st.write(f"🎯 {combined_score:.3f}")

                    # Детали схожести
                    with st.expander("Подробнее о схожести"):
                        st.write(f"Схожесть по жанрам: {genre_score:.3f}")
                        if movie['avg_rating'] > 0:
                            st.write(f"Вклад рейтинга: {rating_score:.3f}")
                        st.write(f"Общий score: {combined_score:.3f}")

                st.divider()

# Дополнительные функции
st.markdown("---")
st.subheader("🔍 Поиск фильмов по жанру")

search_genre = st.text_input("Введите жанр для поиска:")
if search_genre:
    genre_movies = movies[movies['genres'].str.contains(search_genre, case=False)]
    if len(genre_movies) > 0:
        st.write(f"Найдено фильмов: {len(genre_movies)}")
        st.write(genre_movies[['title', 'genres']].head(10))
    else:
        st.warning("Фильмы с таким жанром не найдены")