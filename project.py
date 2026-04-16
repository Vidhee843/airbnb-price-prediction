import pandas as pd

df=pd.read_csv('combined.csv')

df=df.drop(columns=['id','source','name','description',
       'neighborhood_overview', 'host_id','host_name','host_about',
       'neighbourhood_group_cleansed','bathrooms_text',
       'calendar_last_scraped','license','property_type',
       'minimum_nights','maximum_nights','minimum_minimum_nights',
       'maximum_minimum_nights', 'minimum_maximum_nights',
       'maximum_maximum_nights', 'minimum_nights_avg_ntm',
       'maximum_nights_avg_ntm',
       'has_availability', 'availability_30','availability_60', 
       'availability_90','availability_eoy',
       'first_review','last_review','number_of_reviews_ltm',
       'number_of_reviews_l30d', 'reviews_per_month', 'number_of_reviews_ly',
       'estimated_revenue_l365d',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes','host_listings_count',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms'])

'''drop unrelated variable'''
#'id','source','name','description','neighborhood_overview', 'host_id',
#'host_name','host_about',
#'neighbourhood_group_cleansed','bathrooms_text',
#'calendar_last_scraped','license'
#'minimum_nights','maximum_nights','minimum_minimum_nights',
#'maximum_minimum_nights', 'minimum_maximum_nights',
#'maximum_maximum_nights', 'minimum_nights_avg_ntm',
#'maximum_nights_avg_ntm',


'''room_type cover'''
#property_type


'''availability_365 and instant_bookable cover'''
#'has_availability', 'availability_30','availability_60', 'availability_90'


'''number_of_reviews cover'''
#'first_review','last_review''number_of_reviews_ltm',
#'number_of_reviews_l30d', 'reviews_per_month', 'number_of_reviews_ly',
#'estimated_revenue_l365d',


'''host_total_listings_count cover'''
#'calculated_host_listings_count','calculated_host_listings_count_entire_homes',
#'host_listings_count','calculated_host_listings_count_private_rooms',
#'calculated_host_listings_count_shared_rooms'





'''Using latitude and longitude to calculate distance from airbnb to downtown'''

import numpy as np

def calculate_distance(city_lat, city_lon, data_lat, data_lon):
    d_lat = city_lat - data_lat
    d_lon = city_lon - data_lon
    return np.sqrt(d_lat**2 + d_lon**2) * 111 #calculate distance using pythagon and convert to km unit by *111



#Using Old State House coordinates as Boston downtown location
df_bos=df.loc[df['host_location']=='Boston, MA'].copy()

bos_lat = 42.3601
bos_lon = -71.0589
df_bos['distance_km'] = calculate_distance(bos_lat,bos_lon,df_bos['latitude'],df_bos['longitude'])




#Using The Loop coordinates as Chicago downtown location
df_chi=df.loc[df['host_location']=='Chicago, IL'].copy()

chi_lat = 41.8781
chi_lon = -87.6298
df_chi['distance_km'] = calculate_distance(chi_lat,chi_lon,df_chi['latitude'],df_chi['longitude'])




#Using Civic Center/Hayes Valley coordinates as San Francisco downtown location
df_san=df.loc[df['host_location']=='San Francisco, CA'].copy()

san_lat = 37.7749 
san_lon = -122.4194
df_san['distance_km'] = calculate_distance(san_lat,san_lon,df_san['latitude'],df_san['longitude'])




#Using Time Square coordinates as New York downtown location
df_new=df.loc[df['host_location']=='New York, NY'].copy()

new_lat = 40.7580
new_lon = -73.9851
df_new['distance_km'] = calculate_distance(new_lat,new_lon,df_new['latitude'],df_new['longitude'])


#combine
df=pd.concat([df_bos, df_chi, df_san, df_new], ignore_index=True)

#Too standardize the data, location will be evaluated by distance to downtown
#city or neighbor characteristics will be eliminated
#distance_km cover host_location,neighbourhood_cleansed,latitude,longitude'

df=df.drop(columns=['host_location','neighbourhood_cleansed','latitude','longitude'])




'''Calculate number of active year of the host'''
df['last_scraped'] = pd.to_datetime(df['last_scraped'])
df['host_since'] = pd.to_datetime(df['host_since'])

df['years_active'] = df['last_scraped'].dt.year - df['host_since'].dt.year

df=df.drop(columns=['last_scraped','host_since'])





'''host response data adjustment'''
#convert all % string to int
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '').astype(float)

#all response rate that <= 35% will be considered not response in columns host_response_time
df.loc[df['host_response_rate'] <= 35, 'host_response_time'] = 'not response'





'''amenities will be evaluated by number of amenities'''
import ast
df['amenities_count'] = df['amenities'].apply(lambda x: len(ast.literal_eval(x)))


'''host_verifications is evaluated from level 1-3'''
df['host_verification_level'] = df['host_verifications'].apply(lambda x: len(ast.literal_eval(x)))

df=df.drop(columns=['host_verifications','amenities'])


'''Convert string t,f to 0 and 1'''

df['host_is_superhost'] = df['host_is_superhost'].astype(object)
df.loc[df['host_is_superhost'] == 't', 'host_is_superhost'] = True
df.loc[df['host_is_superhost'] == 'f', 'host_is_superhost'] = False

df['host_identity_verified'] = df['host_identity_verified'].astype(object)
df.loc[df['host_identity_verified'] == 't', 'host_identity_verified'] = True
df.loc[df['host_identity_verified'] == 'f', 'host_identity_verified'] = False

df['instant_bookable'] = df['instant_bookable'].astype(object)
df.loc[df['instant_bookable'] == 't', 'instant_bookable'] = True
df.loc[df['instant_bookable'] == 'f', 'instant_bookable'] = False


df=df.dropna()
df=pd.get_dummies(df, drop_first=False)
df.to_csv('final_data.csv', index=False)





'''Forward/Backward selection'''

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

df.columns
y=df['price']
x=df.drop(columns='price')


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

lr=LinearRegression()
lr.fit(x_train, y_train)

##evaluate RMSE (testing)
y_pred=lr.predict(x_test)
root_mean_squared_error(y_test,y_pred)#119.27

##evaluate RMSE (training)
y_pred=lr.predict(x_train)
root_mean_squared_error(y_train,y_pred)#134.67


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs = SFS(lr, 
          k_features=(1,38), 
          forward=True, 
          scoring='neg_root_mean_squared_error',
          cv=5)

sfs.fit(x_train, y_train)
sfs.k_feature_names_ 
'''
'host_total_listings_count',
 'bathrooms',
 'bedrooms',
 'beds',
 'availability_365',
 'number_of_reviews',
 'estimated_occupancy_l365d',
 'review_scores_cleanliness',
 'review_scores_communication',
 'review_scores_location',
 'review_scores_value',
 'distance_km',
 'host_response_time_not response',
 'host_identity_verified_False',
 'room_type_Entire home/apt',
 'room_type_Hotel room',
 'room_type_Shared room',
 'bathroom type_Private',
 'bathroom type_shared'
 '''

X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

lr.fit(X_train_sfs, y_train)


##evaluate RMSE (testing)
y_pred = lr.predict(X_test_sfs)
root_mean_squared_error(y_test, y_pred)#119.06

##evaluate RMSE (training)
y_pred = lr.predict(X_train_sfs)
root_mean_squared_error(y_train, y_pred)#135.09




sfs = SFS(lr, 
          k_features=(1,38), 
          forward=False, 
          scoring='neg_root_mean_squared_error',
          cv=5)

sfs.fit(x_train, y_train)
sfs.k_feature_names_ 
'''
'host_total_listings_count',
 'bathrooms',
 'bedrooms',
 'beds',
 'availability_365',
 'number_of_reviews',
 'estimated_occupancy_l365d',
 'review_scores_cleanliness',
 'review_scores_checkin',
 'review_scores_location',
 'review_scores_value',
 'distance_km',
 'host_response_time_not response',
 'host_identity_verified_False',
 'room_type_Entire home/apt',
 'room_type_Hotel room',
 'room_type_Shared room',
 'bathroom type_shared'
 '''

X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

lr.fit(X_train_sfs, y_train)


##evaluate RMSE (testing)
y_pred = lr.predict(X_test_sfs)
root_mean_squared_error(y_test, y_pred)#119.22

##evaluate RMSE (training)
y_pred = lr.predict(X_train_sfs)
root_mean_squared_error(y_train, y_pred)#135.16






'''Random Forest'''

from sklearn.ensemble import RandomForestRegressor

rfg = RandomForestRegressor(random_state=1)
rfg.fit(x_train,y_train)


#training
y_pred=rfg.predict(x_train)
root_mean_squared_error(y_train, y_pred)##47.01


#testing
y_pred=rfg.predict(x_test)
root_mean_squared_error(y_test, y_pred)#100.67


parameter_grid = {'n_estimators':[50,100,150,200,250,300],
    'max_depth': range(1,15),'min_samples_split':range(2,6)}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(rfg,parameter_grid,verbose=3,scoring="neg_root_mean_squared_error",cv=5)

grid.fit(x_train,y_train)
grid.best_params_
#{'max_depth': 14, 'min_samples_split': 4, 'n_estimators': 200}


rfg = RandomForestRegressor(max_depth=14, min_samples_split=4, n_estimators=200,random_state=1)
rfg.fit(x_train,y_train)

#training
y_pred=rfg.predict(x_train)
root_mean_squared_error(y_train, y_pred)##58.62


#testing
y_pred=rfg.predict(x_test)
root_mean_squared_error(y_test, y_pred)#99.81


top_feature = pd.DataFrame({
    'feature': x_train.columns,
    'importance': rfg.feature_importances_  # or dt.feature_importances_
}).sort_values('importance', ascending=False)

print(top_feature)
#top 5: bathrooms, bedrooms, amenities count, accommodate, distance

































