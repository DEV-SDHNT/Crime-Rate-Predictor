import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('./Datasets/crime_dataset_india.csv')

cities={'Chennai':12053697,
        'Ludhiana':1988438,
        'Pune' :7345848,
        'Delhi' :33807403,
        'Mumbai' :21673149,
        'Surat' :8330528,
        'Visakhapatnam':2385110,
        'Bangalore' :14008262,
        'Kolkata' :155707786,
        'Ghaziabad' :1055190,
        'Hyderabad' :11068877,
        'Jaipur' :4308510,
        'Lucknow' :4038214,
        'Bhopal':2624865,
        'Patna' :2633243,
        'Kanpur' :3289142,
        'Varanasi' :1789047,
        'Nagpur' :3106340,
        'Meerut' :1835403,
        'Ahmedabad' :8854444,
        'Thane':1261517,
        'Indore' :3393380,
        'Rajkot' :2096981,
        'Vasai' :1423890,
        'Agra' :2422342,
        'Kalyan' :1770000,
        'Nashik' :2294299,
        'Srinagar' :1737502,
        'Faridabad':1220229}

#df['Crime Description']=df['Crime Description'].inverse_transform()
#relatable=df.groupby('Crime Description')['Crime Rate'].mean().reset_index()
#
#fig.px.line(
#    relatable,
#    x="Crime Description",
#    y='Crime Rate',
#    markers=True,
#    title="Relation",
#    labels={'Crime rate','Average Crime Rate'}
    #)
#fig.show()
