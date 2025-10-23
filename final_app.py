import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка всех данных
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    links = pd.read_csv('links.csv')  # Для IMDb ID
    return movies, ratings, links


movies, ratings, links = load_data()

# Расчет рейтингов
movie_stats = ratings.groupby('movieId').agg({
    'rating': ['mean', 'count']
}).round(2)
movie_stats.columns = ['rating_mean', 'rating_count']
movies_with_stats = movies.merge(movie_stats, on='movieId').merge(links, on='movieId')


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_with_stats['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = create_model()


# Функция рекомендаций
def get_recommendations(movie_title, n_recommendations=5):
    try:
        idx = movies_with_stats[movies_with_stats['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_with_stats.iloc[movie_indices]
    except:
        return None


# Интерфейс
st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")

st.title("🎬 Ultimate Movie Recommender")
st.markdown("---")

# Поиск фильмов
search_term = st.text_input("🔍 Поиск фильма по названию:")
if search_term:
    filtered_movies = movies_with_stats[movies_with_stats['title'].str.contains(search_term, case=False)]
    if len(filtered_movies) > 0:
        st.write(f"Найдено фильмов: {len(filtered_movies)}")

        # Быстрый выбор из найденных
        selected_from_search = st.selectbox("Выберите из найденных:", filtered_movies['title'].values)
        selected_movie = selected_from_search
    else:
        st.warning("Фильмы не найдены")
        selected_movie = st.selectbox("Выберите фильм:", movies_with_stats['title'].values)
else:
    selected_movie = st.selectbox("Выберите фильм:", movies_with_stats['title'].values)

# Основные рекомендации
if selected_movie:
    movie_data = movies_with_stats[movies_with_stats['title'] == selected_movie].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Выбранный фильм")
        st.write(f"**{movie_data['title']}**")
        st.write(f"**Жанры:** {movie_data['genres']}")
        st.write(f"**Рейтинг:** ⭐ {movie_data['rating_mean']:.1f}/5")
        st.write(f"**Оценок:** {movie_data['rating_count']}")

        # Ссылка на IMDb
        if pd.notna(movie_data['imdbId']):
            imdb_url = f"https://www.imdb.com/title/tt{movie_data['imdbId']:07d}/"
            st.markdown(f"[📺 Открыть в IMDb]({imdb_url})")

    with col2:
        if st.button("🎯 Получить рекомендации", type="primary"):
            recommendations = get_recommendations(selected_movie, 5)

            if recommendations is not None:
                st.subheader("💡 Вам может понравиться:")

                for i, movie in recommendations.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{movie['title']}**")
                            st.write(f"_{movie['genres']}_")
                        with col2:
                            st.write(f"⭐ {movie['rating_mean']:.1f}")
                        with col3:
                            st.write(f"👥 {movie['rating_count']}")

                    st.divider()
            else:
                st.error("Не удалось найти рекомендации для этого фильма")

# Дополнительные функции
st.markdown("---")
st.subheader("📊 Статистика базы данных")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего фильмов", len(movies))
with col2:
    st.metric("Всего оценок", len(ratings))
with col3:
    st.metric("Средний рейтинг", f"{ratings['rating'].mean():.1f}/5")