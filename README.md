# Business Overview
Operating in a competitive hospitality market, Royal hotel chain serves a diverse clientele with its City Hotels and Resort Hotels. 
Each property type offers tailored services but also contends with unique challenges such as fluctuating demand, cancellation rates 
and evolving customer preferences. The business seeks to enhance profitability while maintaining high service standards through 
data-driven operational improvements.

# Data Source
hotel_booking.csv file, containing detailed information about the hotel's performance over the years.
https://www.kaggle.com/datasets/saadharoon27/hotel-booking-dataset

# Tool
Python

# Data Wrangling/Preparation

- Data loading and Inspection
- Handling null values
- Data cleaning and transforming
- Exploratory Data Analysis
### Exploring the hotel booking dataset to answer key business questions:
- Location of guests
- Pay Per Night
- Price per night over the year
- Distribution of nights spent by market segment and hotel type
- Analyzing preference of guests
- Analyzing special request by customers
- Busy months for the hotel
- Length of stay at the hotel
- Bookings by market segment
- Total cancelled bookings
- Month with the highest number of cancellations

# Data Analysis
## Using Python
### Importing Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px

### Importing Data, Data Wrangling and Visualization:
df = pd.read_csv('Hotel_bookings.csv')
Checking the data:
df.head()
df.shape
df.columns
df.isnull().values.any()
df.isnull().sum()
df.fillna(0, inplace = True)

Resort = cleaned[(cleaned['hotel'] == 'Resort Hotel') & (cleaned['is_canceled'] == 0)]
City = cleaned[(cleaned['hotel'] == 'City Hotel') & (cleaned['is_canceled'] == 0)]

trace = go.Pie(
    labels=labels,
    values=values,
    hoverinfo='label+percent',
    textinfo='value'
)

guests = px.choropleth(country_wise, locations = country_wise['country'],
                       color = country_wise['No of guests'],
                       hover_name = country_wise['country'],
                       title = "Country of Guests")

plt.figure(figsize = (12, 8))
sns.boxplot(x='reserved_room_type', y='adr', hue='hotel', data=cleaned2)
plt.title("Price of Room Types Per Night")
plt.xlabel("Room Type")
plt.show()

px.line(final, x = 'month', y = ['price_resort', 'price_city']
        , title = 'Room Price Per Night Over the Month')

plt.figure(figsize = (12,8))
sns.boxplot(x='market_segment', 
             y='stays_in_weekend_nights', 
             hue='hotel', 
             data=cleaned)
plt.title("Stays in Weekend Nights by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Stays in Weekend Nights")
plt.show()

px.pie(cleaned, names = cleaned['meal'].value_counts().index,
       values = cleaned['meal'].value_counts().values, hole = 0.5)

sns.countplot(x='total_of_special_requests', data =cleaned)
plt.title("Total Special Requests")
plt.show()

sns.countplot(x='total_of_special_requests', data =cleaned, hue = 'hotel')
plt.title("Total Special Requests")
plt.show()

px.line(data_frame=final_busy, x='month', y=['no of guests in resort', 'no of guests in city'], title='Total No Of Guests Per Month')

cleaned_data['market_segment'].value_counts()

fig = px.pie(cleaned_data, names='market_segment', title="Booking Per Market Segment")
fig.update_traces(rotation=90, textinfo="percent + label")
fig.show()

### Summary Statistics
- import pandas as pd
- cleaned_data = pd.read_csv("hotel_bookings.csv") 
- print(cleaned_data.describe())

### Model Building:
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('hotel_bookings.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

df = pd.get_dummies(df, drop_first=True)

X = df.drop('adr', axis=1)
y = df['adr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled).fit()
print(model.summary())

y_pred = model.predict(X_test_scaled)
r_squared = model.rsquared
print(f'R-squared on the test set: {r_squared}')

from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

### Model Result:
       # OLS Regression Results                            

R-squared:                       0.587
Adj. R-squared:                  0.581
F-statistic:                     115.3
Prob (F-statistic):               0.00                                      

# Insights

- Bookings show a steady decline from August to January, with the highest in August (8624) and the lowest in January (4115). This indicates a strong seasonal pattern affecting occupancy rates.
- The hotel heavily depends on Online Travel Agents for a majority of its bookings.
- The highest number of bookings comes from Online Travel Agents (OTA) (35,673), followed by Offline TA/TO (15,880) and Direct bookings (10,648). This highlights a strong dependence on OTAs for bookings.
- While most room assignments are correct (104,414), there is a notable number of incorrect assignments (14,796), which could impact guest satisfaction.
- The vast majority of repeated guests (65,769) do not make special requests, while only a small number (2,360) do
- The variation in ADR across room types indicates an opportunity to refine pricing strategies.

# Recommendations

- Review the value proposition of Resort Hotel to ensure it justifies the higher ADR. Enhance amenities and guest experiences to increase perceived value.
- Conduct a comprehensive review of the pricing strategy for different room types. Utilize data analytics to understand market demand and adjust prices accordingly.
- Conduct a thorough analysis to identify the factors contributing to the revenue decline in 2017. This could involve examining market conditions, competition, customer feedback, and changes in marketing strategies.
- Promote the availability of special requests to repeated guests through direct communication channels such as emails, mobile apps and during check-in. 
- Develop targeted marketing campaigns and promotional offers specifically for weekend stays. These could include discounted rates, special event packages, or themed weekends.
- Investigate the reasons behind the high number of booking changes for room types 'A' and 'B'. This could involve analyzing guest feedback and ensuring that these room types meet guest expectations.
