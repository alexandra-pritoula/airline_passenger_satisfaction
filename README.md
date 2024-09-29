<h1 align="center">Airline Passenger Satisfaction Project</h1>

<div align="center">
  
![GitHub last commit](https://img.shields.io/github/last-commit/alexandra-pritoula/airline_passenger_satisfaction?style=for-the-badge&logo=github) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/alexandra-pritoula/airline_passenger_satisfaction?style=for-the-badge) ![Github contributors](https://img.shields.io/github/contributors/alexandra-pritoula/airline_passenger_satisfaction?style=for-the-badge&color=D580FF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo)](https://opensource.org/licenses/MIT) ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

</div>

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/alexandra-pritoula/README">
    <img src=Images/runway_airplane.jpg alt="Logo" width="300" height="300">
  </a>

<!-- ABOUT THE PROJECT -->
## About The Project

The aim of this project is to determine the main factors affecting airline passenger's satisfaction levels using a customer survey data set. Numerous Machine Learning models were trained, evaluated and compared. The best model was fine-tuned to predict new passenger satisfaction levels. 

<!-- INSTALLATION AND SETUP -->
## Installation and Setup

In order to replicate the results of this project please use the software and dependencies presented below:

### Codes and Resources Used

  - **Editor:** VS Code, version 1.92.1 (Universal)
  - **Python:** version 3.12.4

### Python Packages Used

  - **General Purpose:** `pickle`
  - **Data Manipulation:** `pandas`, `numpy`,
  - **Data Visualisation:** `seaborn`, `matplotlib.pyplot`
  - **Statistics (scipy.stats):** `chi2_contingency`, `iqr`
  - **Machine Learning (sklearn):** `model_selection.cross_val_score`, `model_selection.KFold`, `model_selection.train_test_split`, `model_selection.RandomizedSearchCV`, `model_selection.GridSearchCV`, `preprocessing.StandardScaler`, `tree.DecisionTreeClassifier`, `neighbors.KNeighborsClassifier`, `ensemble.GradientBoostingClassifier`, `ensemble.RandomForestClassifier`

<!-- DATA -->
## Data Set

The original data set can be found through the following [link](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction). However, it was downloaded from this [site](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data), where it was cleaned and preprocessed for classification. Further preprocessing has been carried out in this project to optimise machine learning methods. 

| FEATURE  | DESCRIPTION | PREPROCESSING |
| --------- | -------- | -------- |
| ***Gender*** | Gender of the passengers (Female, Male) | Converted to categorical |
| ***Customer Type*** | The customer type (Loyal customer, disloyal customer) | Capitalised and converted to categorical |
| ***Age*** | The actual age of the passengers | None |
| ***Type of Travel*** | Purpose of the flight of the passengers (Personal Travel, Business Travel) | Capitalised and converted to categorical |
| ***Class*** | Travel class in the plane of the passengers (Business, Eco, Eco Plus) | Converted to categorical and set new ordered categoried as 1<2<3 | 
| ***Flight distance*** | The flight distance of this journey | None |
| ***Inflight wifi service*** | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5) | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created|
| ***Departure/Arrival time convenient*** | Satisfaction level of Departure/Arrival time convenient | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created, imputed missing values using the mode|
| ***Ease of Online booking*** | Satisfaction level of online booking | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Gate location*** | Satisfaction level of Gate location | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Food and drink*** | Satisfaction level of Food and drink | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Online boarding*** | Satisfaction level of online boarding | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Seat comfort*** | Satisfaction level of Seat comfort | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Inflight entertainment*** | Satisfaction level of inflight entertainment | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***On-board service*** | Satisfaction level of On-board service | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Leg room service*** | Satisfaction level of Leg room service | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Baggage handling*** | Satisfaction level of baggage handling | Converted to categorical and set new ordered categoried as 1<2<3<4<5 |
| ***Check-in service*** | Satisfaction level of Check-in service | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Inflight service*** | Satisfaction level of inflight service | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Cleanliness*** | Satisfaction level of Cleanliness | Converted to categorical and set new ordered categoried as 1<2<3<4<5, NaN'a were created |
| ***Departure Delay in Minutes*** | Minutes delayed when departure | None |
| ***Arrival Delay in Minutes*** | Minutes delayed when Arrival | None |
| *Satisfaction* | Airline satisfaction level(Satisfaction, neutral or dissatisfaction) | Converted to categorical and set new ordered categoried as 1<2 |

Other preprocessing included capitalising all column names. Dropping the created NaN values for features with less than 5% missing values. The variable `Departure/Arrival time convenient` had more than the 5% threshold of missing values, so the mode was used to impute its values. Columns `Gender`, `Customer Type`, and `Type of Travel` were One-Hot Encoded. In addition, the data was scaled to improve the machine learning models. 

Three additional features were created in an attempt to improve predictive performance. 

| FEATURE  | DESCRIPTION | PREPROCESSING |
| --------- | -------- | -------- |
| ***Total Delay*** | The sum of `Departure Delay in Minutes` and `Arrival Delay in Minutes` | Set as categorical and split into 3 ordered categories 1<2<3 |
| ***Age Group*** | Binned the `Age` feature into 5 ordered categories | One-hot encoded |
| ***Overall Service Satisfaction*** | The mean of the service related features for each passenger | Rounded |

Only the `Overall Service Satisfaction` variable has been kept in the final model. 

<!-- CODE STRUCTURE -->
## Code Structure

```bash
├── Data
│   ├── Preprocessed_1
│   │   ├── train_preprocessed_1.csv
│   │   └── test_preprocessed_1.csv
│   ├── Preprocessed_2
│   │   ├── train_preprocessed_2.csv
│   │   └── test_preprocessed_2.csv
│   ├── Preprocessed_3
│   │   ├── train_preprocessed_3.csv
│   │   └── test_preprocessed_3.csv
│   ├── Preprocessed_4
│   │   ├── train_preprocessed_4.csv
│   │   └── test_preprocessed_4.csv
│   └── Raw
│   │   ├── train.csv
│   │   └── test.csv
├── Images
│   └── runway_airplane.jpg
├── Models
│   ├── Model_1
│   │   ├── bp_data.csv
│   │   ├── decision_tree.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── knn.pkl
│   │   └── random_forest.pkl
│   ├── Model_2
│   │   ├── bp_data_2.csv
│   │   ├── decision_tree_2.pkl
│   │   ├── gradient_boosting_2.pkl
│   │   ├── knn_2.pkl
│   │   └── random_forest_2.pkl
│   ├── Model_3
│   │   ├── bp_data_3.csv
│   │   ├── decision_tree_3.pkl
│   │   ├── gradient_boosting_3.pkl
│   │   ├── knn_3.pkl
│   │   └── random_forest_3.pkl
│   ├── Model_4
│   │   ├── bp_data_4.csv
│   │   ├── decision_tree_4.pkl
│   │   ├── gradient_boosting_4.pkl
│   │   ├── knn_4.pkl
│   │   └── random_forest_4.pkl
│   ├── Tuned_Models
│   │   ├── rf_rscv_model.pkl
│   │   └── rf_gscv_model.pkl
├── notebooks
│   ├── 1_data_preparation
│   ├── 2_exploratory_data_analysis_and_preprocessing
│   ├── 3_feature_engineering
│   └── 4_model_training_and_evaluation
├── references
├── reports
│   └── figures

```
<!-- RESULTS -->
## Results




<!-- LICENSE -->
## License

Distributed under the MIT License.
