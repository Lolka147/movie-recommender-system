import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка данных
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')  # Новый файл!
    return movies, ratings


movies, ratings = load_data()

# Расчет средних рейтингов
movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
movies_with_ratings = movies.merge(movie_ratings, on='movieId')


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_with_ratings['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, tfidf


cosine_sim, tfidf = create_model()

# Интерфейс
st.set_page_config(page_title="Рекомендации фильмов", layout="wide")

st.title("🎬 Умная система рекомендации фильмов")
st.write("Найдите фильмы, которые вам понравятся!")

col1, col2 = st.columns([1, 2])

with col1:
    selected_movie = st.selectbox("Выберите фильм:", movies_with_ratings['title'].values)

    if st.button("🎯 Найти похожие фильмы", type="primary"):
        st.session_state.show_recommendations = True

with col2:
    if selected_movie:
        movie_data = movies_with_ratings[movies_with_ratings['title'] == selected_movie].iloc[0]
        st.subheader(movie_data['title'])
        st.write(f"**Жанры:** {movie_data['genres']}")
        st.write(f"**Средний рейтинг:** ⭐ {movie_data['rating']:.1f}/5")

if st.session_state.get('show_recommendations', False):
    st.subheader("🎭 Рекомендуемые фильмы:")

    idx = movies_with_ratings[movies_with_ratings['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    for i, (movie_idx, score) in enumerate(sim_scores, 1):
        movie = movies_with_ratings.iloc[movie_idx]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i}. {movie['title']}**")
            st.write(f"Жанры: {movie['genres']}")
        with col2:
            st.write(f"⭐ {movie['rating']:.1f}")
            st.write(f"Схожесть: {score:.2f}")

        st.divider()