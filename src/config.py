''' In this code we define the configuration variables
that are needed throughout this project'''

# for dataset
OUTPUT_FEATURE = 'Outcome'
ORIGINAL_RAW_DATASET = '../input/diabetes.csv'

RAW_TRAIN_DATASET = '../raw_train_set.pickle'
RAW_TEST_DATASET = '../input/raw_test_set.pickle'

CLEAN_TRAIN_DATASET = '../input/clean_train_set.pickle'
CLEAN_TEST_DATASET = '../input/clean_test_set.pickle'

# for plots
NULL_CHECK_HEATMAP = '../plots/null_check_heatmap.png'

# for model training purposes
DISTANCE_METRIC = 'cosine'
NUM_FOLDS = 5
BASELINE_MODEL = '../models/KNN_BASELINE_'