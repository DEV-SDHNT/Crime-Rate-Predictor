import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor,plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset_path = './Datasets/crime_dataset_india.csv'
df = pd.read_csv(dataset_path)

# print(df.columns.to_list())  List column names from the dataset
# Convert Date and Time columns
df['year'] = pd.to_datetime(df['Time of Occurrence'],format='mixed').dt.year
df['City_Name']=df['City']
print(df['year'].unique())
df['Crime_Description']=df['Crime Description']
# Encode categorical features
label_encoder = LabelEncoder()
df['City'] = label_encoder.fit_transform(df['City'])
df['Crime Description'] = label_encoder.fit_transform(df['Crime Description'])
df['Weapon Used'] = label_encoder.fit_transform(df['Weapon Used'])
#df['Crime Domain'] = label_encoder.fit_transform(df['Crime Domain'])


# Get unique cities from dataset
unique_cities = df['City_Name'].unique()

# Approximate city population data (example values, replace with real data)
#city_population = {city: np.random.randint(500000, 20000000) for city in unique_cities}  # Generate random population data
city_population={'Chennai':12053697,
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

df['Population'] = df['City_Name'].map(city_population)
df['Crime Rate'] = ((df.groupby('City_Name')['Crime Code'].transform('count') / df['Population']) * 100000).round(1)
#df.dropna(subset=['Crime Rate'], inplace=True)  # Remove cities without population data

# Selecting features and target
features = ['City','year','Crime Code','Crime Description']
target = 'Crime Rate'

X = df[features]
# X=df.drop(columns=['Crime Rate'])

y = df[target]
# X=pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

scaler=StandardScaler()
xtrain_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
# XGBoost Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(xtrain_scaled, y_train)

# Best Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, RMSE: {rmse}')

joblib.dump(best_model,"./Model/xgbModel.pkl")
df.to_csv('./Datasets/processedDataset.csv')


#import plotly.express as px
#yearCrime=df.groupby('Crime Domain')['Crime Rate'].mean().reset_index()
#fig=px.line(
#    yearCrime,
#    x="Crime Domain",
#    y="Crime Rate",
#    markers=True,
#    title='Crime rate Over years',
#    labels={'Crime Rate','Average Crime Rate'}
#    )
#fig.show()

#plt.figure(figsize=(12,8))
#plot_importance(best_model,max_num_features=10,importance_type='gain')
#plt.title("Features")
#plt.tight_layout()
#plt.show()

