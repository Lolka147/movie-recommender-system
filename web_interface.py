import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Загрузка данных БЕЗ заголовков
@st.cache_data
def load_data():
    # Указываем что в файле нет заголовков и задаем названия столбцов
    movies = pd.read_csv('movies.csv', header=None, names=['movieId', 'title', 'genres'])

    # ОЧИСТКА ДАННЫХ:
    # 1. Убираем точку с запятой в конце жанров
    movies['genres'] = movies['genres'].str.rstrip(';')

    # 2. Заменяем пустые строки на "Unknown"
    movies['genres'] = movies['genres'].fillna('Unknown')

    # 3. Убираем строки где genres пустое или NaN
    movies = movies[movies['genres'].notna()]
    movies = movies[movies['genres'] != '']

    # 4. Убираем строки где title пустое или NaN
    movies = movies[movies['title'].notna()]
    movies = movies[movies['title'] != '']

    # Сбрасываем индексы после фильтрации
    movies = movies.reset_index(drop=True)

    return movies


movies = load_data()

# Покажем информацию о данных
st.title("🎬 Система рекомендации фильмов")
st.write(f"В базе {len(movies)} фильмов после очистки")

# Покажем немного данных для проверки
st.write("**Первые 5 фильмов в базе:**")
st.write(movies[['title', 'genres']].head())

# Проверим есть ли пустые значения
st.write("**Проверка данных:**")
st.write(f"Пустых жанров: {movies['genres'].isna().sum()}")
st.write(f"Пустых названий: {movies['title'].isna().sum()}")


# Создание модели
@st.cache_data
def create_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


try:
    cosine_sim = create_model()
    st.success("✅ Модель успешно создана!")

    # Выбор фильма
    selected_movie = st.selectbox("Выберите фильм:", movies['title'].values)

    if st.button("Найти рекомендации"):
        # Поиск рекомендаций
        idx = movies[movies['title'] == selected_movie].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Берем топ-5 (исключаем сам фильм)
        movie_indices = [i[0] for i in sim_scores]

        # Вывод результатов
        st.subheader("🎭 Рекомендуемые фильмы:")
        for i, movie_idx in enumerate(movie_indices, 1):
            movie = movies.iloc[movie_idx]
            st.write(f"{i}. **{movie['title']}**")
            st.write(f"   🎭 Жанры: {movie['genres']}")
            st.write(f"   📊 Схожесть: {cosine_sim[idx][movie_idx]:.3f}")

except Exception as e:
    st.error(f"❌ Ошибка при создании модели: {e}")
    st.write("**Отладочная информация:**")
    st.write(f"Тип данных в genres: {type(movies['genres'].iloc[0])}")
    st.write(f"Пример жанров: {movies['genres'].iloc[:5].tolist()}")