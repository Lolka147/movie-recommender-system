import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings


movies, ratings = load_data()


# Функция для Content-Based рекомендаций
def content_based_recommendations(movie_title, n_recommendations=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices]


# Функция для популярных фильмов
def popular_recommendations(n_recommendations=5):
    movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_counts = ratings.groupby('movieId')['rating'].count().reset_index()

    movie_stats = movie_ratings.merge(movie_counts, on='movieId')
    movie_stats = movie_stats.merge(movies, on='movieId')

    # Фильтруем фильмы с достаточным количеством оценок
    popular_movies = movie_stats[movie_stats['rating_y'] > 50]
    popular_movies = popular_movies.sort_values(['rating_x', 'rating_y'], ascending=[False, False])

    return popular_movies.head(n_recommendations)


# Интерфейс
st.title("🎬 Гибридная система рекомендаций")

tab1, tab2, tab3 = st.tabs(["По жанрам", "Популярные", "Сравнение"])

with tab1:
    st.subheader("Рекомендации по схожести жанров")
    selected_movie = st.selectbox("Выберите фильм:", movies['title'].values)

    if st.button("Найти похожие"):
        recommendations = content_based_recommendations(selected_movie)
        st.write("Фильмы с похожими жанрами:")
        for i, movie in recommendations.iterrows():
            st.write(f"- **{movie['title']}** ({movie['genres']})")

with tab2:
    st.subheader("Самые популярные фильмы")
    n_movies = st.slider("Количество фильмов:", 3, 10, 5)

    if st.button("Показать популярные"):
        popular = popular_recommendations(n_movies)
        st.write("Самые популярные фильмы:")
        for i, movie in popular.iterrows():
            st.write(f"- **{movie['title']}** ⭐ {movie['rating_x']:.1f} ({movie['rating_y']} оценок)")

with tab3:
    st.subheader("Сравнение подходов")
    selected_movie = st.selectbox("Выберите фильм для сравнения:", movies['title'].values, key="compare")

    if st.button("Сравнить рекомендации"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**По жанрам:**")
            content_recs = content_based_recommendations(selected_movie)
            for i, movie in content_recs.iterrows():
                st.write(f"- {movie['title']}")

        with col2:
            st.write("**Популярные:**")
            popular_recs = popular_recommendations(5)
            for i, movie in popular_recs.iterrows():
                st.write(f"- {movie['title']} ⭐ {movie['rating_x']:.1f}")