<h1 align="center">Airline Passenger Satisfaction Project (work-in-progress)</h1>

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

The aim of this project is to determine the main factors affecting airline passenger's satisfaction levels using a customer survey data set. Numerous standard machine learning models and neural networks were trained, evaluated and compared. The best model, a deep neural network, has been fine-tuned to predict new passenger satisfaction levels. 

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
| ***Arrival Delay in Minutes*** | Minutes delayed when Arrival | Imputed missing values |
| *Satisfaction* | Airline satisfaction level(Satisfaction, neutral or dissatisfaction) | Converted to categorical and set new ordered categoried as 1<2 |

Other preprocessing included capitalising all column names, dropping the created NaN values for features with less than 5% missing values. The variable `Departure/Arrival time convenient` had more than the 5% threshold of missing values, so the mode was used to impute its values. Columns `Gender`, `Customer Type`, and `Type of Travel` were One-Hot Encoded. In addition, the data was scaled to improve the machine learning models. The performance of the models were tested on different combinations of these preprocessing steps.


Two additional features were created in an attempt to improve predictive performance. 

| FEATURE  | DESCRIPTION | PREPROCESSING |
| --------- | -------- | -------- |
| ***Overall Service Satisfaction*** | The mean of the service related features for each passenger | Rounded |
| ***Premium Service*** | A variable indicating whether the overall service was premium | Categorised `Overall Service Satisfaction` into 2 groups, `1` and `2`  |


<!-- CODE STRUCTURE -->
## Code Structure

```bash

Airline Passenger Satisfaction Project
├── Data
├── Images
│   └── runway_airplane.jpg
├── Models
└── Notebooks
    ├── data_preparation.ipynb
    ├── exploratory_data_analysis.ipynb
    ├── feature_engineering.ipynb
    └── model_training_and_evaluation.ipynb

```
<!-- RESULTS -->
## Results
WIP

<!-- REFERENCES -->
## References
1.  Learn Statistics Easily, Kendall Tau-b vs Spearman: Which Correlation Coefficient Wins?, Learn Statistics Easily, 4 Jan 2024. [Link](https://statisticseasily.com/kendall-tau-b-vs-spearman/#) 

2. Minitab Support, What are Concordant and Discordant Pairs?, Minitab Support, 2024. [Link](https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/tables/supporting-topics/other-statistics-and-tests/what-are-concordant-and-discordant-pairs/)

3. Shaun Turney, Chi-Square Test of Independence | Formula, Guide and Examples, *Scribbr*, June 22, 2023. [Link](https://www.scribbr.com/statistics/chi-square-test-of-independence/#:~:text=A%20chi%2Dsquare%20test%20of%20independence%20works%20by%20comparing%20the,values%20of%20the%20other%20variable.&text=Example%3A%20Expected%20values%20The%20city,frequencies%20using%20the%20contingency%20table)
4. Aurelian Géron, Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd Edition), O'Reilly Media Inc, 20 January 2023.
 
5. Haldun Akoglu, User's Guide to Correlation Coefficients, *National Library of Medicine*, 7 August 2018. [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6107969/#:~:text=Cramer's%20V%20is%20an%20alternative,Cramer's%20V%20(Table%202).)

6. Scikit-learn developers, IterativeImputer, *Scikit-Learn*, 2024. [Link](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)

7. Kyaw Saw Htoon, A Guide to KNN Imputation, *Medium*, 3 July 2020. [Link](https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e)
   
8. Hannah Igboke, Iterative Imputer for Missing values in Machine Learning, *Medium*, 10 June 2024. [Link](https://medium.com/learning-data/iterative-imputer-for-missing-values-in-machine-learning-32bd8b5b697a#:~:text=Statistical%20models%20used%20in%20iterative%20imputation&text=DecisionTreeRegressor%3A%20non%2Dlinear%20regression%20models,no%20need%20for%20feature%20scaling.)
   
<!-- LICENSE -->
## License

Distributed under the MIT License.
