# Diabetes Prediction
***

Building a predictive model to identify which patient has diabetes. Using the [Pima Indians Diabetes dataset.](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) from [Kaggle](https://www.kaggle.com/). I use K-Nearest Neighbours model to solve the problem.

The repository has multiple directories, with each serving a different purpose:
- input/: contains:
    - original raw dataset
    - raw dataset split into train and test data.
    - cleaned/feature engineered version of train and test data
- notebooks/: Consists of one jupyter notebook. It was used for EDA purpose and also experiment with some functions used for feature engineering.
- src/: this directory consists of the source code for the project.
    - config.py: consists of variables which are used all across the code.
    - create_folds.py: used for implementing stratified kfold cross validation.
    - feature_engg.py: used for cleaning the dataset and applying feature engineering techniques.
    - test_functionalities: using pytest module, i define some data sanity checks on the training data.
    - train.py: this file contains the code for implementing the model. The train and the inference stage.
 - results/: contains the train and validation accuracy scores for k:1-50. This helps me identify the ideal k value.
 - plots/: used to store all the plots for visualizing the data and interpreting experimental results.

## To obtain clean data and split it into train and test set, use the following command:
  ```python train.py --clean dataset```
  
### To train the model using Stratified Kfold cross validation use following command:
  ```python train.py --train skfold```

### For inference stage, use:
  ```python train.py --test inference```

### For more information use:
  ```python train.py --help```
