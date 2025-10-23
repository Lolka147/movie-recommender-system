import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка данных
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    return movies


movies = load_data()


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = create_model()

# Интерфейс
st.title("🎬 Система рекомендации фильмов")
st.write(f"В базе {len(movies)} фильмов")

# Выбор фильма
selected_movie = st.selectbox("Выберите фильм:", movies['title'].values)

if st.button("Найти рекомендации"):
    # Поиск рекомендаций
    idx = movies[movies['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]

    # Вывод результатов
    st.subheader("Рекомендуемые фильмы:")
    for i, movie_idx in enumerate(movie_indices, 1):
        movie = movies.iloc[movie_idx]
        st.write(f"{i}. **{movie['title']}**")
        st.write(f"   Жанры: {movie['genres']}")