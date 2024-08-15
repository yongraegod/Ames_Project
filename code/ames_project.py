# !pip install folium
import pandas as pd
import folium
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

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
--------------------------------------------
# 집값으로 시각화

# 집값의 최소값과 최대값 구하기
min_price = df['Sale_Price'].min()
max_price = df['Sale_Price'].max()

for _, row in df.iterrows():
    # 집값을 0에서 1 사이의 값으로 정규화
    normalized_price = (row['Sale_Price'] - min_price) / (max_price - min_price)
    
    # 정규화된 값으로 색상 구하기 (파란색 -> 빨간색)
    color = plt.cm.coolwarm(normalized_price)
    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
    
    # 지도에 원 추가
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color_hex,
        fill=True,
        fill_color=color_hex,
        fill_opacity=0.7,
        popup=f"Price: ${row['Sale_Price']:,.0f}\nNeighborhood: {row['Neighborhood']}"
    ).add_to(ames_map)

# 결과를 HTML 파일로 저장
ames_map.save('maps/ames_house_price_map.html')
------------------------------------------------
import pandas as pd
import folium
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
# Ames 데이터 불러오기
df = pd.read_csv("../Ames_Project/data/houseprice-with-lonlat.csv")

# 가격 구간별 집 갯수
# Define the bin edges for the price ranges (0 to max_price with 50,000 intervals)
bins = np.arange(0, df['Sale_Price'].max() + 50000, 50000)

# Calculate the number of houses in each price range
price_counts = pd.cut(df['Sale_Price'], bins=bins).value_counts().sort_index()

# Adjusting the tick labels to display more neatly and highlighting the 10~15만 range
plt.figure(figsize=(10, 6))
colors = ['skyblue'] * len(price_counts)  # Default color for all bars

# Finding the index for the 100,000 to 150,000 range
highlight_index = price_counts.index.get_loc(pd.Interval(100000, 150000, closed='right'))
colors[highlight_index] = 'red'  # Change color for 10~15만 range

# Create the bar plot with specified colors
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
price_counts.plot(kind='bar', color=colors)
plt.title('가격 구간별 집의 갯수')
plt.xlabel('가격 구간 ($)')
plt.ylabel('집의 갯수')
plt.xticks(rotation=60, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
