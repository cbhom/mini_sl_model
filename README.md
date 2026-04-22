(This readme is a text-only version of the Project Report file)
Flight Delay Prediction using Supervised Learning 

Mini Project Report
Submitted in partial fulfillment of the requirements for the course
CAI2501 – [Machine Learning Lab]
Submitted by:
Subhom Baruah (Roll No: 20241CAI0098)
Mainuddin T J (Roll No: 20241CAI0097)
Under the guidance of:
Dr.Alamelu Mangai Jothidurai
Department of Computer Science and Engineering
[Presidency University, Bangalore]
April 20261. Introduction
Flight delays are a critical issue in modern aviation systems, affecting operational efficiency, airline profitability, and passenger satisfaction. With increasing air traffic, delays have become more frequent and complex due to multiple contributing factors such as weather, congestion, and operational inefficiencies.
Machine learning provides a powerful approach to analyze historical flight data and predict delays, enabling proactive decision-making. This project focuses on predicting flight delay severity using data-driven techniques.

2. Problem Statement

The objective of this project is:
To predict whether a flight will experience a significant delay (≥ 60 minutes) using historical flight data. 
Unlike basic delay prediction, this project emphasizes delay severity, which is more useful for real-world applications.

3. Dataset Description

The dataset includes real-world flight data with features such as:
Flight timings (DepTime, ArrTime)
Airline information (Airline, Carrier)
Route details (Origin, Destination)
Delay metrics (ArrDelay, DepDelay)
Delay causes:
CarrierDelay
WeatherDelay
NASDelay
LateAircraftDelay
Key Observation:

The dataset contains only flights delayed by 15 minutes or more, which influenced the modeling strategy.

4. Exploratory Data Analysis (EDA)

EDA was performed to uncover patterns and relationships:
Correlation analysis:

Strong correlation between:
Departure Delay → Arrival Delay
Time-based patterns:

Day-of-week variation:

Weather impact:

Weather significantly increases delay duration.


5. Data Preprocessing

Steps performed:
Removed rows with missing critical values (ArrDelay)
Filled missing delay cause values with 0
Dropped irrelevant columns:
FlightNum
TailNum
CancellationCode
Diverted
Reason

To improve data quality and reduce noise.


6. Feature Engineering

New features were created:
DepHour: Extracted hour from departure time
is_peak_hour: 1 if departure between 16–22
is_weekend: 1 for weekends
total_cause_delay: Sum of all delay causes
Purpose
To capture hidden patterns and improve model performance.




7. Target Variable

A new target variable was defined:
0 → Delay < 60 minutes
1 → Delay ≥ 60 minutes
This converts the problem into a binary classification task.


8. Model Development

A machine learning pipeline was implemented:
Preprocessing
StandardScaler → numerical features
OneHotEncoder → categorical features
Models Used
Random Forest Classifier (primary)
Logistic Regression (baseline)
Why Random Forest?
Handles non-linear relationships
Robust to noise
Works well with mixed data


9. Model Evaluation

Metrics used:
Accuracy
Classification Report (Precision, Recall, F1-score)
Confusion Matrix
Observation
Random Forest performed better than Logistic Regression due to its ability to capture complex relationships.


10. Key Insights

Departure delays strongly impact arrival delays
Weather is a major contributor to severe delays
Peak hours increase congestion
Combined delay causes improve prediction


11. Conclusion

The project successfully developed a machine learning model to predict flight delay severity.
Achievements
Built structured ML pipeline
Identified key delay factors
Achieved reliable predictions

12. Future Work

Use real-time flight data
Apply advanced models (XGBoost)
Deploy using Streamlit
Improve feature engineering

Research Papers

Paper 1: Flight Delay Prediction Using Machine Learning
Objective: Predict delays using historical flight data
Methodology: Random Forest, Logistic Regression
Findings: Weather and departure delay are key factors
Relevance: Supports model choice and feature engineering

Paper 2: Predicting Flight Delays Using Data Mining Techniques
Objective: Identify delay patterns
Methodology: Decision Trees, SVM
Findings: Airline and route significantly affect delays
Relevance: Validates use of categorical features

Paper 3: Airline Delay Analysis and Prediction
Objective: Analyze causes of delays
Methodology: Statistical + ML methods
Findings: Late aircraft is a dominant factor
Relevance: Supports total_cause_delay feature

Paper 4: Machine Learning Models for Air Traffic Delay Prediction
Objective: Compare ML models
Methodology: Random Forest, Gradient Boosting
Findings: Ensemble models perform best
Relevance: Justifies Random Forest usage

Paper 5: Impact of Weather on Flight Delays
Objective: Study weather effects
Methodology: Regression analysis
Findings: Weather significantly increases delays
Relevance: Supports weather-based insights

Paper 6: Predicting Flight Delays with Big Data
Objective: Use large-scale data
Methodology: Distributed ML models
Findings: Large datasets improve accuracy
Relevance: Highlights scalability

Paper 7: Delay Propagation in Airline Networks
Objective: Study cascading delays
Methodology: Network analysis
Findings: Delays propagate across flights
Relevance: Supports time-based features

Paper 8: Flight Delay Classification Using Ensemble Learning
Objective: Improve classification accuracy
Methodology: Ensemble techniques
Findings: Ensemble models outperform single models
Relevance: Supports Random Forest

Paper 9: Aviation Delay Prediction Using Neural Networks
Objective: Apply deep learning
Methodology: Neural networks
Findings: High accuracy but complex
Relevance: Future improvement direction

Paper 10: Data-Driven Approaches for Flight Delay Prediction
Objective: Use data analytics
Methodology: ML + statistical methods
Findings: Feature engineering is critical
Relevance: Validates project approach
