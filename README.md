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

The aim of this project is to determine the main factors affecting airline passenger's satisfaction levels using a customer survey data set. The data has been downloaded from Kaggle and preprocessed for classification. Numerous standard machine learning models and neural networks have been trained, evaluated and compared. The best model, a deep neural network, has been extensively fine-tuned (both manually and with a grid search) and then used to predict new passenger satisfaction levels.  

<!-- INSTALLATION AND SETUP -->
## Installation and Setup

To replicate the results of this project, please use the software and dependencies presented below:

### Codes and Resources Used

  - **Editor:** VS Code, version 1.92.1 (Universal)
  - **Python:** version 3.12.4

### Python Packages Used

  - **General Purpose:** `pickle`, `os`, `itertools`, `pprint`
  - **Data Manipulation:** `pandas`, `numpy`
  - **Data Visualisation:** `seaborn`, `matplotlib.pyplot`, `matplotlib.patches.Patch`
  - **Statistics (scipy.stats):** `chi2_contingency`, `iqr`, `contingency.association`
  - **Machine Learning (sklearn):** `model_selection.cross_val_score`, `model_selection.KFold`, `model_selection.train_test_split`, `model_selection.RandomizedSearchCV`, `model_selection.GridSearchCV`, `model_selection.cross_validate`, `tree.DecisionTreeClassifier`, `neighbors.KNeighborsClassifier`, `ensemble.GradientBoostingClassifier`, `ensemble.RandomForestClassifier`, `confusion_matrix`, `ConfusionMatrixDisplay`, `preprocessing.StandardScaler`, `preprocessing.LabelEncoder`, `preprocessing.MinMaxScaler`, `discriminant_analysis.LinearDiscriminantAnalysis`, `linear_model.LogisticRegression`, `metrics.precision_score`, `metrics.recall_score`, `metrics.make_scorer`, `metrics.accuracy_score`, `metrics.roc_curve`, `metrics.auc`, `metrics.roc_auc_score`, `impute.IterativeImputer`, `impute.KNNImputer`
  - **Deep Learning (torch):** `nn`, `nn.init`, `nn.BCELoss`, `optim`, `utils.data.TensorDataset`, `utils.data.DataLoader`, `torchmetrics`

<!-- DATA -->
## Data Set

The original data set can be found through the following [link](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction). However, it has been downloaded from this [site](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data), where it was cleaned and preprocessed for classification. Additional preprocessing has been done in this project to optimise machine learning methods. 

| FEATURE  | DESCRIPTION | PREPROCESSING |
| --------- | -------- | -------- |
| ***Gender*** | Gender of the passengers (Female, Male) | Converted to categorical |
| ***Customer Type*** | The customer type (Loyal customer, disloyal customer) | Capitalised and converted to categorical |
| ***Age*** | The actual age of the passengers | None |
| ***Type of Travel*** | Purpose of the flight of the passengers (Personal Travel, Business Travel) | Capitalised and converted to categorical|
| ***Class*** | Travel class in the plane of the passengers (Business, Eco, Eco Plus) | Converted to categorical and set new ordered categories as 1<2<3 | 
| ***Flight distance*** | The flight distance of this journey | None |
| ***Inflight wifi service*** | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5) | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created|
| ***Departure/Arrival time convenient*** | Satisfaction level of Departure/Arrival time convenient | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created, imputed missing values using the mode|
| ***Ease of Online booking*** | Satisfaction level of online booking | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Gate location*** | Satisfaction level of Gate location | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Food and drink*** | Satisfaction level of Food and drink | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Online boarding*** | Satisfaction level of online boarding | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Seat comfort*** | Satisfaction level of Seat comfort | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Inflight entertainment*** | Satisfaction level of inflight entertainment | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***On-board service*** | Satisfaction level of On-board service | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Leg room service*** | Satisfaction level of Leg room service | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Baggage handling*** | Satisfaction level of baggage handling | Converted to categorical and set new ordered categories as 1<2<3<4<5 |
| ***Check-in service*** | Satisfaction level of Check-in service | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Inflight service*** | Satisfaction level of inflight service | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Cleanliness*** | Satisfaction level of Cleanliness | Converted to categorical and set new ordered categories as 1<2<3<4<5, NaN'a were created |
| ***Departure Delay in Minutes*** | Minutes delayed when departure | None |
| ***Arrival Delay in Minutes*** | Minutes delayed when Arrival | Imputed missing values |
| *Satisfaction* | Airline satisfaction level(Satisfaction, neutral or dissatisfaction) | Converted to categorical and set new ordered categories as 1<2 |

Other preprocessing included capitalising all column names and dropping the created NaN values for features with less than 5% missing values. The variable `Departure/Arrival time convenient` had more than the 5% threshold of missing values, so the mode was used to impute its values. Columns `Gender`, `Customer Type`, and `Type of Travel` were One-Hot Encoded. In addition, the data was scaled to improve the machine-learning models. The performance of the models was tested using different combinations of these preprocessing steps.


Two additional features were created in an attempt to improve predictive performance. 

| FEATURE  | DESCRIPTION | PREPROCESSING |
| --------- | -------- | -------- |
| ***Overall Service Satisfaction*** | The mean of the service-related features for each passenger | Rounded |
| ***Premium Service*** | A variable indicating whether the overall service was premium | Categorised `Overall Service Satisfaction` into two groups, `1` and `2`  |


<!-- CODE STRUCTURE -->
## Code Structure

```bash

Airline Passenger Satisfaction Project
├── Data
│   ├── Feature_Extraction
│   │   └── train_oss.pkl
│   ├── Feature_Selection
│   │   └── train_fs.pkl
│   ├── Predictions
│   │   ├── prediction.csv
│   │   └── labels.csv
│   ├── Preprocessed
│   │   ├── non_applicable_imputed.pkl
│   │   ├── test_preprocessed.pkl
│   │   ├── train_imputed_iterative.pkl
│   │   ├── train_imputed_knn.pkl
│   │   ├── train_preprocessed_2.pkl
│   │   ├── train_preprocessed_3.pkl
│   │   ├── train_preprocessed_dropna_mode.pkl
│   │   └── train_preprocessed.pkl
│   ├── Raw
│   │   ├── test.csv
│   │   └── train.csv
├── Images
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── count_plot_2.png
│   ├── count_plot.png
│   ├── pairplot.png
│   └── runway_airplane.jpg
├── LICENSE.txt
├── Models
│   ├── Imputation_Model_1
│   │   └── results_iterative_model.pkl
│   ├── Imputation_Model_2
│   │   └── results_knn_model.pkl
│   ├── Imputation_Model_3
│   │   └── results_dropna_mode_model.pkl
│   ├── Imputation_Model_4
│   │   ├── model_rf.pkl
│   │   └── results_nonapplicable_mode_model.pkl
│   ├── Model_fe
│   │   └── results_fe_model.pkl
│   ├── Model_fs
│   │   └── results_fs_model.pkl
│   └── Neural_Nets
│   │   ├── best_model_1.pth
│   │   ⋮
│   │   ├── best_model_70.pth
│   │   ├── test_losses_1.pkl
│   │   ⋮
│   │   ├── test_losses_70.pkl
│   │   ├── train_accuracies_1.pkl
│   │   ⋮
│   │   ├── train_accuracies_70.pkl
│   │   ├── training_losses_1.pkl
│   │   ⋮
│   │   ├── training_losses_70.pkl
│   │   ├── validation_accuracies_1.pkl
│   │   ⋮
│   │   └── validation_accuracies_70.pkl
├── Notebooks
│   ├── data_preparation.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   ├── final_model_analysis_summary.ipynb
│   └── model_training_and_evaluation.ipynb
└── README.md

```
<!-- RESULTS -->
## Results

The EDA section of this project showed that premium customers are generally satisfied with the airline company, while economy customers are divided. To ensure customer satisfaction, the company should improve economy-class experiences and services. The top three features driving customer satisfaction are `Online Boarding`, `Inflight Wi-Fi Service`, and `Personal Travel` (as seen in the feature importance plot). 

The original data set has been processed in several ways and compared to find the one leading to the best performance. The best-performing preprocessed data encodes ‘Non-Applicable’ services as a separate category. Numerous machine learning models were then implemented, including LDA, KNN, Logistic Regression and Random Forest; however, a Neural Network implemented using PyTorch and manually hyper-tuned achieved the best results.  

The final model was trained with one hundred epochs, and its loss and accuracy curves depicted good model performance and stability. The validation accuracy of this model was 0.96636.  The final model's predictions on the test data set achieved an accuracy score of 0.96705, a precision of 0.97436, and a recall of 0.94993, which is very close to those on the validation set. The consistent results confirm that the model is stable and performs well on new data. The AUC score under the ROC curve was 0.96518, a very high result. 

The extensive preprocessing and hyperparameter tuning did not substantially improve the results compared to the first random forest base model, which had an accuracy of  0.964429, a precision of 0.957307, and a recall of 0.980958. Given the base model's already high accuracy and precision, one has to wonder whether the extremely time-consuming effort would have been financially interesting to the airline. 

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

10. James Gareth, Daniela Witten, Trevor Hastie and Robert Tibshirani, An Introduction to Statistical Learning, Springer, New York, 2015.

11. PyTorch Contributors, BCEWithLogitsLoss, Pytorch, 2023, [(link)](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

12. Pytorch Contributors, SGD, Pytorch, 2023, [(link)](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
   
<!-- LICENSE -->
## License

Distributed under the MIT License.
