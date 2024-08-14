!pip install folium
import pandas as pd
import folium
import plotly.express as px
import numpy as np

# Ames 데이터 불러오기
df = pd.read_csv("../Ames_Project/data/houseprice-with-lonlat.csv")

# ames 중심 위도, 경도 구하기
df['Longitude'].mean()
df['Latitude'].mean()

# 흰 도화지 맵 그리기
ames_map = folium.Map(location = [42.034, -93.642],
                    zoom_start = 12,
                    tiles='cartodbpositron')
ames_map.save('maps/ames_map.html')
