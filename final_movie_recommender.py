import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go


# Загрузка всех данных
@st.cache_data
def load_data():
    # Фильмы
    movies = pd.read_csv('movies.csv', header=None, names=['movieId', 'title', 'genres'])
    movies['genres'] = movies['genres'].str.rstrip(';').fillna('Unknown')
    movies = movies.dropna()

    # Рейтинги
    try:
        ratings = pd.read_csv('ratings.csv')
        movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        movies = movies.merge(movie_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(0)
        movies['rating_count'] = movies['rating_count'].fillna(0)
    except:
        movies['avg_rating'] = 0
        movies['rating_count'] = 0

    return movies, ratings if 'ratings' in locals() else None


movies, ratings = load_data()


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = create_model()


# Функции рекомендаций
def content_based_recommendations(movie_title, top_n=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    return [(i, score) for i, score in sim_scores]


def popular_recommendations(top_n=5):
    popular = movies[movies['rating_count'] > 10].sort_values('avg_rating', ascending=False)
    return popular.head(top_n)


def hybrid_recommendations(movie_title, top_n=5):
    content_recs = content_based_recommendations(movie_title, top_n * 2)

    # Добавляем рейтинги к рекомендациям
    enhanced = []
    for idx, score in content_recs:
        movie = movies.iloc[idx]
        rating_bonus = movie['avg_rating'] / 5.0 if movie['avg_rating'] > 0 else 0.5
        final_score = score * 0.7 + rating_bonus * 0.3
        enhanced.append((idx, final_score, score, rating_bonus))

    enhanced.sort(key=lambda x: x[1], reverse=True)
    return enhanced[:top_n]


# ИНТЕРФЕЙС
st.set_page_config(page_title="Ultimate Movie Recommender", layout="wide", page_icon="🎬")

# Главный заголовок
st.title("🎬 Ultimate Movie Recommendation System")
st.markdown("---")

# Сайдбар
with st.sidebar:
    st.header("🎛️ Панель управления")

    st.subheader("Выбор режима")
    recommendation_mode = st.radio(
        "Тип рекомендаций:",
        ["Гибридные 🚀", "По жанрам 🎭", "Популярные ⭐"]
    )

    st.subheader("Настройки")
    top_n = st.slider("Количество рекомендаций", 3, 15, 8)

    st.subheader("Статистика базы")
    st.metric("Всего фильмов", len(movies))
    if ratings is not None:
        st.metric("Всего оценок", f"{len(ratings):,}")
        st.metric("Пользователей", ratings['userId'].nunique())

# Основная панель
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Рекомендации", "📊 Аналитика", "🔍 Поиск", "ℹ️ О проекте"])

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Выбор фильма")
        selected_movie = st.selectbox("Выберите фильм:", movies['title'].values)

        if selected_movie:
            movie_data = movies[movies['title'] == selected_movie].iloc[0]

            st.info("**Информация о фильме:**")
            st.write(f"**🎭 Жанры:** {movie_data['genres']}")
            if movie_data['avg_rating'] > 0:
                st.write(f"**⭐ Рейтинг:** {movie_data['avg_rating']:.1f}/5")
                st.write(f"**👥 Оценок:** {int(movie_data['rating_count'])}")

    with col2:
        if st.button("🎬 Получить рекомендации", type="primary", use_container_width=True):
            with st.spinner("Анализируем ваши предпочтения..."):

                if recommendation_mode == "Гибридные 🚀":
                    recommendations = hybrid_recommendations(selected_movie, top_n)
                    st.success("🚀 Гибридные рекомендации (жанры + рейтинги)")

                elif recommendation_mode == "По жанрам 🎭":
                    recommendations = content_based_recommendations(selected_movie, top_n)
                    st.success("🎭 Рекомендации по схожести жанров")

                else:  # Популярные
                    popular_movies = popular_recommendations(top_n)
                    st.success("⭐ Самые популярные фильмы")

                # Отображение результатов
                if recommendation_mode != "Популярные ⭐":
                    for i, (idx, *scores) in enumerate(recommendations, 1):
                        movie = movies.iloc[idx]
                        with st.container():
                            cols = st.columns([3, 1, 1, 1])
                            with cols[0]:
                                st.write(f"**{i}. {movie['title']}**")
                                st.caption(f"_{movie['genres']}_")
                            with cols[1]:
                                if movie['avg_rating'] > 0:
                                    st.metric("Рейтинг", f"{movie['avg_rating']:.1f}")
                                else:
                                    st.write("⭐ --")
                            with cols[2]:
                                if len(scores) > 1:
                                    st.metric("Схожесть", f"{scores[1]:.3f}")
                            with cols[3]:
                                if len(scores) > 0:
                                    st.metric("Score", f"{scores[0]:.3f}")
                            st.divider()
                else:
                    for i, (_, movie) in enumerate(popular_movies.iterrows(), 1):
                        with st.container():
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.write(f"**{i}. {movie['title']}**")
                                st.caption(f"_{movie['genres']}_")
                            with cols[1]:
                                st.metric("Рейтинг", f"{movie['avg_rating']:.1f}")
                            with cols[2]:
                                st.metric("Оценок", int(movie['rating_count']))
                            st.divider()

with tab2:
    st.subheader("📈 Аналитика базы данных")

    col1, col2 = st.columns(2)

    with col1:
        # Распределение жанров
        st.write("**🎭 Топ жанров:**")
        all_genres = '|'.join(movies['genres']).split('|')
        genre_counts = pd.Series(all_genres).value_counts().head(10)

        fig_genres = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            title="Самые популярные жанры"
        )
        st.plotly_chart(fig_genres, use_container_width=True)

    with col2:
        # Распределение рейтингов
        if ratings is not None:
            st.write("**📊 Распределение оценок:**")
            rating_dist = ratings['rating'].value_counts().sort_index()

            fig_ratings = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Распределение пользовательских оценок"
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

with tab3:
    st.subheader("🔍 Расширенный поиск")

    col1, col2 = st.columns(2)

    with col1:
        search_type = st.radio("Тип поиска:", ["По названию", "По жанру"])

        search_results = pd.DataFrame()  # Инициализируем пустой DataFrame

        if search_type == "По названию":
            search_term = st.text_input("Введите название фильма:")
            if search_term:
                search_results = movies[movies['title'].str.contains(search_term, case=False)]
        else:
            genre_term = st.text_input("Введите жанр:")
            if genre_term:
                search_results = movies[movies['genres'].str.contains(genre_term, case=False)]

    with col2:
        if not search_results.empty:
            st.write(f"**Найдено фильмов: {len(search_results)}**")
            for _, movie in search_results.head(10).iterrows():
                st.write(f"• **{movie['title']}**")
                st.caption(f"Жанры: {movie['genres']}")
                if movie['avg_rating'] > 0:
                    st.caption(f"Рейтинг: ⭐ {movie['avg_rating']:.1f}")
        elif st.session_state.get('search_performed', False):
            st.warning("Фильмы не найдены")

    # Добавляем кнопку поиска
    if st.button("🔍 Выполнить поиск", key="search_btn"):
        st.session_state.search_performed = True
        st.rerun()

with tab4:
    st.subheader("ℹ️ О системе рекомендаций")

    st.write("""
    ### 🎯 Как работает система:

    **Гибридные рекомендации 🚀**
    - Анализирует схожесть жанров (Content-Based Filtering)
    - Учитывает рейтинги пользователей
    - Комбинирует оба подхода для лучших результатов

    **Рекомендации по жанрам 🎭**
    - Находит фильмы с похожими жанрами
    - Использует косинусную меру для вычисления схожести

    **Популярные фильмы ⭐**
    - Показывает фильмы с высокими рейтингами
    - Учитывает количество оценок для достоверности

    ### 🛠 Технологии:
    - Python, Pandas, Scikit-learn
    - Streamlit для веб-интерфейса
    - Plotly для визуализации данных
    """)

    st.success("🎓 Курсовой проект по теме 'Система рекомендаций фильмов'")

# Футер
st.markdown("---")
st.caption("🎬 Ultimate Movie Recommender | Курсовой проект | 2024")