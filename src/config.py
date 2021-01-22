''' In this code we define the configuration variables
that are needed throughout this project'''

# for dataset
OUTPUT_FEATURE = 'Outcome'
KFOLD_COLUMN_NAME = 'kfold'
FILE_DELIMITER = ','
ENCODING_TYPE = 'UTF-8'
ORIGINAL_RAW_DATASET = '../input/diabetes.csv'
DATASET_SHAPE = (768,9)

RAW_TRAIN_DATASET = '../input/raw_train_set.pickle'
RAW_TEST_DATASET = '../input/raw_test_set.pickle'

CLEAN_TRAIN_DATASET = '../input/clean_train_set.pickle'
CLEAN_TEST_DATASET = '../input/clean_test_set.pickle'

# for plots
NULL_CHECK_HEATMAP = '../plots/null_check_heatmap.png'

# for model training purposes
DISTANCE_METRIC = 'cosine'
NUM_FOLDS = 5
NUM_NEIGHBOURS = 15     # obtained from empirical evidence
#knn is a lazy learner, so no need to save the model
TRAIN_ACCURACY_SCORE = '../results/train_accuracy_score.pickle'
VALIDATION_ACCURACY_SCORE = '../results/validation_accuracy_score.pickle'
