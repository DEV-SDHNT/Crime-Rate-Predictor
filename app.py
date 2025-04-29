from flask import Flask, render_template,request
from flask_caching import Cache
import pandas as pd
import folium
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
import joblib
import shutil
import os
import time
import plotly.express as px

app=Flask(__name__)

cache=Cache(app,config={'CACHE_TYPE':'simple'})

model=joblib.load('./Model/xgbModel.pkl')
df=pd.read_csv('./Datasets/processedDataset.csv')
features = ['City', 'Crime Code', 'year','Crime Description']


# Interactive Map with Crime Rate Visualization
@cache.memoize(timeout=6000)
def generate_map():
    if os.path.exists("./templates/crime_map.html"):
        return "templates/crime_map.html"
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
    city_crime['Color'] = city_crime['Crime Rate'].apply(lambda x: 'red' if x > (max_crime * 0.6) else 'green')
    # Create Folium Map
    map_center = [city_crime['Latitude'].mean(), city_crime['Longitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=5,tiles='https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}{r}.png',attr='Stadiamaps')
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
    print("Map Generated.")
    return 'templates/crime_map.html'

@cache.memoize(timeout=6000)
def crimeRateDistribution(df):
    fig=px.histogram(df,x='Crime Rate',title="Crime Rate Distribution",template='plotly_dark')
    fig.write_html("templates/crimeRateDistribution.html")
    print("Crime Rate Distribution Graph created")
    shutil.copy('templates/crimeRateDistribution.html','static/crimeRateDistribution.html')
    print("Crime Rate Distribution Graph Done !!")
    return "templates/crimeRateDistribution.html"

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
        color_continuous_scale='Reds',
        template='plotly_dark'
    )
    print("Top 10 Done")
    fig.write_html("templates/TopCrimeHotspot.html")
    shutil.copy('templates/TopCrimeHotspot.html','static/TopCrimeHotspot.html')
    return "templates/TopCrimeHotspot.html"

cache.memoize(timeout=6000)
def CrimeAndCrimeRate(df):
    if 'Crime_Description' not in df.columns or 'Crime Rate' not in df.columns:
        return "Crime description or crime rate missing"
    relation=df.groupby('Crime_Description')['Crime Rate'].mean().reset_index()
    fig=px.line(
        relation,
        x="Crime_Description",
        y="Crime Rate",
        markers=True,
        title="Crime Rate Vs Crime Domain",
        labels={'Crime Rate','Average Crime Rate'},
        template="plotly_dark"
    )
    print("Crime Vs Crime Rate graph Done.")
    fig.write_html("templates/CrimeAndCrimeRate.html")
    shutil.copy("templates/CrimeAndCrimeRate.html","static/CrimeAndCrimeRate.html")
    return "templates/CrimeAndCrimeRate.html"

cache.memoize(timeout=600)
def YearlyRate(df):
    if 'year' not in df.columns or 'Crime Rate' not in df.columns:
        return "Year or Crime Rate missing "
    relation=df.groupby('year')['Crime Rate'].mean().reset_index()
    fig=px.line(
        relation,
        x='year',
        y="Crime Rate",
        markers=True,
        title="Yearly Crime Trend",
        labels={'Crime Rate','Average Crime Rate'},
        template='plotly_dark'
    )
    fig.write_html("templates/YearlyRate.html")
    shutil.copy("templates/YearlyRate.html","static/YearlyRate.html")
    return "templates/YearlyRate.html"

@cache.memoize(timeout=6000)
@app.route('/',methods=['GET','POST'])
def home():
    pred=None
    if request.method=="POST":
        cityName=request.form['city']
        encoder=LabelEncoder()
        encoder.fit(df['City_Name'])
        city=encoder.transform([cityName])[0]

        crime_code=int(request.form['crime_code'])

        year=int(request.form['year'])

        description=request.form['crimedescription']
        crimedescriptionEncode=encoder.fit(df['Crime_Description'])
        crimedescription=encoder.transform([description])[0]
        print(crimedescription)
        input_data=pd.DataFrame([[city,crime_code,year,crimedescription]],columns=features)
        pred=model.predict(input_data)[0]
        print(pred)
    cities=sorted(df['City_Name'].unique())
    crimes=sorted(df['Crime_Description'].unique())
    crimeCode=sorted(df['Crime Code'].unique())
    generate_map()

    graph1=crimeRateDistribution(df)

    graph2=TopCrimeHotSpot(df)

    graph3=CrimeAndCrimeRate(df)

    graph4=YearlyRate(df)

    return render_template(
        'dashboard.html',
        cities=cities,
        crimes=crimes,
        crimeCode=crimeCode,
        prediction=pred,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4
    )

if __name__ == '__main__':
    # port=int(os.environ.get("PORT",5000))
    app.run(debug=True,host="0.0.0.0")
