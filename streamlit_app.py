import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib as plt
import folium
from streamlit_folium import st_folium


st.title("Earthquakes")
st.write(
    "Let's start!"
)

# Завантаження даних з файлу CSV 
data = pd.read_csv('data/earthquake_1995-2023.csv')

# Відображення заголовку
st.title("Карта на основі даних")
# Відображення DataFrame
st.write("Набір даних:")
st.write(data)

# Відображення карти на основі даних про широту і довготу
st.map(data[['latitude', 'longitude']])


# Заголовок додатка
st.title("Інтерактивна мапа землетрусів (1995-2023)")

# Легенда для кольорів
st.markdown("""
### Легенда кольорів магнітуд:
- <span style="color:green;">**Зелений**</span>: Магнітуда < 5  
- <span style="color:orange;">**Помаранчевий**</span>: 5 ≤ Магнітуда < 6  
- <span style="color:red;">**Червоний**</span>: 6 ≤ Магнітуда < 7  
- <span style="color:darkred;">**Темно-червоний**</span>: Магнітуда ≥ 7  
""", unsafe_allow_html=True)

# Налаштування початкової мапи
map_center = [0, 0]  # Центр карти для глобального огляду
m = folium.Map(location=map_center, zoom_start=2)

# Нормалізація магнітуд для маркерів
magnitude_min = data['magnitude'].min()
magnitude_max = data['magnitude'].max()

# Функція для вибору кольору залежно від магнітуди
def get_color(magnitude):
    if magnitude < 5:
        return 'green'
    elif 5 <= magnitude < 6:
        return 'orange'
    elif 6 <= magnitude < 7:
        return 'red'
    else:
        return 'darkred'

# Додавання маркерів на мапу
for _, row in data.iterrows():
    location = [row['latitude'], row['longitude']]
    magnitude = row['magnitude']
    
    # Розмір маркера залежно від магнітуди
    radius = (magnitude - magnitude_min) / (magnitude_max - magnitude_min) * 10 + 3

    # Додавання маркера з кольором та інформацією
    folium.CircleMarker(
        location=location,
        radius=radius,
        color=get_color(magnitude),
        fill=True,
        fill_color=get_color(magnitude),
        fill_opacity=0.7,
        tooltip=f"{row['location']}, Magnitude: {magnitude}"
    ).add_to(m)

# Відображення інтерактивної мапи у Streamlit
st_folium(m, width=700, height=500)


# Графік розподілу магнітуд
st.subheader("Розподіл магнітуд землетрусів")
# Побудова гістограми з Altair
hist = alt.Chart(data).mark_bar().encode(
    alt.X('magnitude:Q', bin=alt.Bin(maxbins=30), title='Магнітуда'),
    alt.Y('count()', title='Кількість землетрусів'),
    tooltip=['count()']
).properties(
    width=600,
    height=400,
    title='Гістограма розподілу магнітуд землетрусів'
)
# Виведення графіку в Streamlit
st.altair_chart(hist, use_container_width=True)


# Графік розподілу магнітуд
st.subheader("Розподіл глибин землетрусів")
# Побудова гістограми з Altair
hist = alt.Chart(data).mark_bar().encode(
    alt.X('depth:Q', bin=alt.Bin(maxbins=30), title='Глибина землетрусу'),
    alt.Y('count()', title='Кількість землетрусів'),
    tooltip=['count()']
).properties(
    width=600,
    height=400,
    title='Гістограма розподілу магнітуд землетрусів'
)
# Виведення графіку в Streamlit
st.altair_chart(hist, use_container_width=True)



# Заголовок додатка
st.title("Аналіз глибини землетрусів (1995-2023)")

# Побудова boxplot для глибини землетрусів
st.subheader("Розподіл глибини землетрусів")

# Створення графіка boxplot з Altair
boxplot = alt.Chart(data).mark_boxplot().encode(
    y=alt.Y('depth:Q', title='Глибина (км)'),
    tooltip=['depth']
).properties(
    width=600,
    height=400,
    title='Boxplot глибини землетрусів'
)

# Виведення графіка у Streamlit
st.altair_chart(boxplot, use_container_width=True)



# Перетворення стовпця часу на формат datetime
data['time'] = pd.to_datetime(data['date_time'])

# Виділення року із дати
data['year'] = data['time'].dt.year



# Перетворення стовпця часу на формат datetime
data['time'] = pd.to_datetime(data['date_time'])

# Виділення року із дати
data['year'] = data['time'].dt.year

# Заголовок додатка
st.title("Кількість землетрусів по роках (1995-2023)")

# Групування даних по роках і сортування за кількістю землетрусів
earthquakes_per_year = (
    data.groupby('year').size().reset_index(name='count').sort_values(by='count', ascending=False)
)

# Виведення таблиці з даними
# st.subheader("Таблиця: Кількість землетрусів по роках")
# st.dataframe(earthquakes_per_year)

# Транспонування таблиці для вертикального вигляду
vertical_table = earthquakes_per_year.T  # Транспонована таблиця

# Виведення транспонованої таблиці
st.subheader("Кількість землетрусів по роках")
st.dataframe(vertical_table)

# Побудова лінійного графіка з Altair
# st.subheader("Графік: Кількість землетрусів по роках")
line_chart = alt.Chart(earthquakes_per_year).mark_line(point=True).encode(
    x=alt.X('year:O', title='Рік'),
    y=alt.Y('count:Q', title='Кількість землетрусів'),
    tooltip=['year', 'count']
).properties(
    width=700,
    height=400,
    #title='Кількість землетрусів по роках'
)

# Виведення графіка у Streamlit
st.altair_chart(line_chart, use_container_width=True)





# Завантаження даних
#data = pd.read_csv('data/earthquake_1995-2023.csv')

# Видалення пропущених значень (якщо є)
data = data.dropna(subset=['latitude', 'longitude', 'magnitude', 'depth'])

# Нормалізація даних для кращої кластеризації
scaler = StandardScaler()
#data_scaled = scaler.fit_transform(data[['latitude', 'longitude', 'depth', 'magnitude']])
data_scaled = scaler.fit_transform(data[['depth', 'magnitude']])


# K-means кластеризація
kmeans = KMeans(n_clusters=3, random_state=0)
data['kmeans_cluster'] = kmeans.fit_predict(data_scaled)

# DBSCAN кластеризація
dbscan = DBSCAN(eps=0.5, min_samples=10)
data['dbscan_cluster'] = dbscan.fit_predict(data_scaled)

# Візуалізація результатів K-means
st.subheader("Кластеризація K-means (4 кластери)")
kmeans_chart = alt.Chart(data).mark_circle(size=60).encode(
    x=alt.X('longitude:Q', title='Довгота'),
    y=alt.Y('latitude:Q', title='Широта'),
    color='kmeans_cluster:N',
    tooltip=['location', 'magnitude', 'depth']
).properties(
    width=600,
    height=400,
    title="Результати кластеризації K-means"
)
st.altair_chart(kmeans_chart, use_container_width=True)

# Візуалізація результатів DBSCAN
st.subheader("Кластеризація DBSCAN")
dbscan_chart = alt.Chart(data).mark_circle(size=60).encode(
    x=alt.X('longitude:Q', title='Довгота'),
    y=alt.Y('latitude:Q', title='Широта'),
    color='dbscan_cluster:N',
    tooltip=['location', 'magnitude', 'depth']
).properties(
    width=600,
    height=400,
    title="Результати кластеризації DBSCAN"
)
st.altair_chart(dbscan_chart, use_container_width=True)

# Оригінальна мапа з землетрусами
st.title("Інтерактивна мапа землетрусів з кластерами")

# Створення інтерактивної мапи
map_center = [0, 0]  # Центр мапи для глобального огляду
m = folium.Map(location=map_center, zoom_start=2)

# Додавання кластерних маркерів на карту
for _, row in data.iterrows():
    location = [row['latitude'], row['longitude']]
    folium.CircleMarker(
        location=location,
        radius=5,
        color='blue' if row['kmeans_cluster'] == -1 else 'green',
        fill=True,
        fill_opacity=0.6,
        tooltip=f"{row['location']} - K-means: {row['kmeans_cluster']} - DBSCAN: {row['dbscan_cluster']}"
    ).add_to(m)

st_folium(m, width=700, height=500)
