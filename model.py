import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor,plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib


# Load dataset
dataset_path = './Datasets/crime_dataset_india.csv'
df = pd.read_csv(dataset_path)

# print(df.columns.to_list())  List column names from the dataset
# Convert Date and Time columns
df['year'] = pd.to_datetime(df['Time of Occurrence'],format='mixed').dt.year
df['City_Name']=df['City']
print(df['year'].unique())

# Encode categorical features
label_encoder = LabelEncoder()
df['City'] = label_encoder.fit_transform(df['City'])
df['Crime Description'] = label_encoder.fit_transform(df['Crime Description'])
df['Weapon Used'] = label_encoder.fit_transform(df['Weapon Used'])
df['Crime Domain'] = label_encoder.fit_transform(df['Crime Domain'])
df['Case Closed'] = label_encoder.fit_transform(df['Case Closed'])

# Get unique cities from dataset
unique_cities = df['City_Name'].unique()

# Approximate city population data (example values, replace with real data)
city_population = {city: np.random.randint(500000, 20000000) for city in unique_cities}  # Generate random population data

df['Population'] = df['City_Name'].map(city_population)
df['Crime Rate'] = ((df.groupby('City_Name')['Crime Code'].transform('count') / df['Population']) * 100000).round(3)
df.dropna(subset=['Crime Rate'], inplace=True)  # Remove cities without population data
# Selecting features and target
features = ['City','Crime Code','year']
target = 'Crime Rate'
X = df[features]
# X=df.drop(columns=['Crime Rate'])
y = df[target]
# X=pd.get_dummies(X)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# XGBoost Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, RMSE: {rmse}')

joblib.dump(best_model,"./Model/xgbModel.pkl")
df.to_csv('./Datasets/processedDataset.csv')
