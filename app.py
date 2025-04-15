from flask import Flask, render_template,request
from flask_caching import Cache
import pandas as pd
import folium
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
import joblib
import shutil
import os
import hashlib
import plotly
import plotly.express as px



app = Flask(__name__)

cache=Cache(app,config={'CACHE_TYPE':'simple'})

model=joblib.load('./Model/xgbModel.pkl')
df=pd.read_csv('./Datasets/processedDataset.csv')
features = ['City', 'Crime Code', 'year']


def get_df_hasd(df):
    return hashlib.md5(pd.util.hash_pandas_object(df,index=True).values).hexdigest()


# Interactive Map with Crime Rate Visualization
@cache.memoize(timeout=6000)
def generate_map():
    # if not os.path.exists('./templates/crime_map.html'):
    print("Generating map....")
    geolocator = Nominatim(user_agent="crime_mapper")
    city_crime = df.groupby('City_Name').agg({'Crime Rate': 'mean'}).reset_index()
    # Fetch coordinates for dataset cities
    city_crime['Coordinates'] = city_crime['City_Name'].apply(lambda city: geolocator.geocode(city, timeout=10))
    city_crime = city_crime.dropna()
    city_crime['Latitude'] = city_crime['Coordinates'].apply(lambda loc: loc.latitude)
    city_crime['Longitude'] = city_crime['Coordinates'].apply(lambda loc: loc.longitude)
    city_crime.drop(columns=['Coordinates'], inplace=True)
    # Define color scale
    max_crime = city_crime['Crime Rate'].max()
    city_crime['Color'] = city_crime['Crime Rate'].apply(lambda x: 'red' if x > (max_crime * 0.2) else 'green')
    # Create Folium Map
    map_center = [city_crime['Latitude'].mean(), city_crime['Longitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=5,tiles='openstreetmap',attr='OpenStreet Maps')
    # Add markers
    for _, row in city_crime.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=row['Color'],
            fill=True,
            fill_color=row['Color'],
            fill_opacity=0.7,
            popup=f"{row['City_Name']}: {row['Crime Rate']} per 100,000 people"
        ).add_to(crime_map)
    # Save map
    crime_map.save("templates/crime_map.html")
    shutil.copy('templates/crime_map.html','static/crime_map.html')
    print("---|  Interactive crime map saved as templates/crime_map.html  |---")
    return 'templates/crime_map.html'

@cache.memoize(timeout=6000)
def crimeRateDistribution(df):
    print("Crime Rate Distribution Graph created")
    fig=px.histogram(df,x='Crime Rate',title="Crime Rate Distribution")
    return fig.to_html(full_html=False)

# def featureImportance(model,features):
#     importance=model.feature_importances_
#     importance_df=pd.DataFrame({
#         'Feature':features,
#         'Importance':importance
#     }).sort_values(by='Importance',ascending=False)
#     fig=px.bar(importance_df,
#         x='Feature',
#         y='Importance',
#         title='Feature Importance for Crime Rate',
#         labels={'Importance':'Relative importance'},
#         color='Importance',
#         color_continuous_scale='blues')
#     print("Bar graph created!")
#     return fig.to_html(full_html=False)

@cache.memoize(timeout=6000)
def TopCrimeHotSpot(df):
    if 'City' not in df.columns or 'Crime Rate' not in df.columns:
        return "Dataset don't have City and Crime rate columns"
    cityCrimeRate=df.groupby('City_Name')['Crime Rate'].mean().reset_index()
    cityCrimeRate=cityCrimeRate.sort_values(by='Crime Rate',ascending=False).head(10)
    fig=px.bar(
        cityCrimeRate,
        x='City_Name',
        y='Crime Rate',
        title='Top 10 Crime Hotspots ',
        color='Crime Rate',
        color_continuous_scale='Reds'
    )
    print("Top 10 Done")
    return fig.to_html(full_html=False)

@app.route('/',methods=['GET','POST'])
def home():
    pred=None
    if request.method=="POST":
        cityName=request.form['city']
        cityEncode=LabelEncoder()
        cityEncode.fit(df['City_Name'])
        city=cityEncode.transform([cityName])[0]
        crime_code=int(request.form['crime_code'])
        year=int(request.form['year'])

        input_data=pd.DataFrame([[city,crime_code,year]],columns=features)
        pred=model.predict(input_data)[0]
        print(pred)
    cities=sorted(df['City_Name'].unique())
    generate_map()
    graph1=crimeRateDistribution(df)
    graph2=TopCrimeHotSpot(df)
    return render_template('dashboard.html',cities=cities,prediction=pred,graph1=graph1,graph2=graph2)

# @app.route('/analysis')
# def analysis():
#     importance=featureImportance(model,features=features)
#     return render_template('analysis.html',graph1=importance)

if __name__ == '__main__':
    app.run(debug=True)