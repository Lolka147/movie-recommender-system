import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Конфигурация страницы
st.set_page_config(
    page_title="Movie Recommender Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Загрузка данных с обработкой ошибок
@st.cache_data
def load_data():
    try:
        # Фильмы
        movies = pd.read_csv('movies.csv', header=None, names=['movieId', 'title', 'genres'])
        movies['genres'] = movies['genres'].str.rstrip(';').fillna('Unknown')
        movies = movies.dropna()

        # Рейтинги
        ratings = pd.read_csv('ratings.csv')
        movies['movieId'] = movies['movieId'].astype(str)
        ratings['movieId'] = ratings['movieId'].astype(str)

        movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

        movies = movies.merge(movie_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(0)
        movies['rating_count'] = movies['rating_count'].fillna(0)

        return movies, True

    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        # Возвращаем демо-данные
        movies = pd.read_csv('movies.csv', header=None, names=['movieId', 'title', 'genres'])
        movies['genres'] = movies['genres'].str.rstrip(';').fillna('Unknown')
        movies = movies.dropna()

        np.random.seed(42)
        movies['avg_rating'] = np.random.uniform(3.5, 4.8, len(movies))
        movies['rating_count'] = np.random.randint(10, 500, len(movies))

        return movies, False


# Загрузка данных
movies, data_loaded = load_data()


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = create_model()


# Функция рекомендаций
def get_recommendations(movie_title, top_n=5):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        return [(i, score) for i, score in sim_scores]
    except:
        return []


# ИНТЕРФЕЙС
st.title("🎬 Movie Recommender Pro")
st.markdown("---")

# Главные колонки
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Выбор фильма")
    selected_movie = st.selectbox("Выберите фильм:", movies['title'].values)

    if selected_movie:
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        st.write(f"**Жанры:** {movie_info['genres']}")
        if movie_info['avg_rating'] > 0:
            st.write(f"**Рейтинг:** ⭐ {movie_info['avg_rating']:.1f}/5")
            st.write(f"**Оценок:** {int(movie_info['rating_count'])}")

with col2:
    if st.button("🎬 Получить рекомендации", type="primary", use_container_width=True):
        recommendations = get_recommendations(selected_movie, 8)

        if recommendations:
            st.subheader("💡 Вам может понравиться:")

            for i, (movie_idx, score) in enumerate(recommendations, 1):
                movie = movies.iloc[movie_idx]

                with st.container():
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.write(f"**{i}. {movie['title']}**")
                        st.write(f"_{movie['genres']}_")
                    with cols[1]:
                        if movie['avg_rating'] > 0:
                            st.write(f"⭐ {movie['avg_rating']:.1f}")
                    with cols[2]:
                        st.write(f"🎯 {score:.3f}")

                st.divider()
        else:
            st.error("Не удалось найти рекомендации")

# Информация о системе
st.markdown("---")
with st.expander("ℹ️ О системе"):
    st.write("""
    **Movie Recommender Pro** - интеллектуальная система рекомендаций фильмов.

    🔧 **Технологии:**
    - Content-Based Filtering
    - Косинусная мера схожести
    - Машинное обучение (Scikit-learn)
    - Веб-интерфейс (Streamlit)

    📊 **Данные:** MovieLens Dataset
    """)

    if not data_loaded:
        st.warning("⚠️ Используются демо-рейтинги (файл ratings.csv не найден)")

st.caption("🎓 Курсовой проект | Система рекомендаций фильмов | 2024")