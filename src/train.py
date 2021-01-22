'''Here we will train and test the model'''

# import modules here
import config
import feature_engg
import create_folds

import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# train function here
def run(X_train, y_train,
        k_neighbours, fold):
    '''

    :param X_train:
    :param y_train:
    :param k_neighbours:
    :param fold:
    :return:
    '''

# test function here
def inference_stage(dataset, model):
    '''

    :param dataset:
    :param model:
    :return:
    '''
    return


# main gateway here
if __name__ == "__main__":

    # commandline arguments here
    parser = argparse.ArgumentParser()

    # adding the arguments
    parser.add_argument('--clean', type=str,
                        help='Provide argument \"--clean dataset\"'
                             ' to obtain clean train and test data.')

    parser.add_argument('--train', type=str,
                        help='Provide argument \"--train skfold\" '
                             'to train the model using Stratified'
                             ' KFold Cross Validation.')

    parser.add_argument('--test', type=str,
                        help='Provide argument \"--test inference\" '
                             'to test the model and obtain performance metrics.')

    args = parser.parse_args()

    # will use this obj to dump and load pickled files
    dl_obj = feature_engg.DumpLoadFile()

    if args.clean == 'dataset':

        fr_obj = feature_engg.FeatureEngineering()
        # load the raw train set
        raw_train_set = dl_obj.load_file(config.RAW_TRAIN_DATASET)

        # load the raw test set
        raw_test_set = dl_obj.load_file(config.RAW_TEST_DATASET)

        # clean the datasets
        X_clean_train_set, y_clean_train_set = fr_obj.cleaning_data(raw_train_set)
        X_clean_test_set, y_clean_test_set = fr_obj.cleaning_data(raw_test_set)

        # dump the clean datasets
        dl_obj.dump_file(config.CLEAN_TRAIN_DATASET, X_clean_train_set, y_clean_train_set)
        dl_obj.dump_file(config.CLEAN_TEST_DATASET, X_clean_test_set, y_clean_test_set)

    elif args.train == 'skfold':

        # load the train dataset
        X_train, y_train = dl_obj.load_file(config.CLEAN_TRAIN_DATASET)
        print(X_train.shape, y_train.shape)

        # we need to maintain the f1 score on average
        # for all 5 folds for all k neighbours
        f1_score_avg = []

        for k in range(1,51):
            f1_score_run = []
            for fold_value in range(config.NUM_FOLDS):
                f1_score_run.append(run(X_train=X_train, y_train=y_train,
                                    k_neighbours=k, fold=fold_value))

            # storing the average test score of 5 runs for each k
            f1_score_avg.append(np.mean(f1_score_run))

        # dump the metric score, as we will use this as a plot

    elif args.test == 'inference':
        pass