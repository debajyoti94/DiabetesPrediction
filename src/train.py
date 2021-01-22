'''Here we will train and test the model'''

# import modules here
import config
import feature_engg
import create_folds

import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, \
                            accuracy_score

# train function here
def run(dataset,
        k_neighbours, fold):
    '''

    :param dataset:
    :param k_neighbours:
    :param fold:
    :return:
    '''
    train_set = dataset[dataset.kfold != fold]
    valid_set = dataset[dataset.kfold == fold]

    # split train set to X and y
    X_train = train_set.drop(config.OUTPUT_FEATURE, axis=1,
                             inplace=False)
    y_train = train_set[config.OUTPUT_FEATURE]

    # split valid set to X and y
    X_valid = valid_set.drop(config.OUTPUT_FEATURE, axis=1,
                             inplace=False)
    y_valid = valid_set[config.OUTPUT_FEATURE]

    # instantiating the knn classifier
    knn = KNeighborsClassifier(n_neighbors=k_neighbours,
                               metric=config.DISTANCE_METRIC)

    # fit the model
    knn.fit(X_train, y_train)
    # get predictions
    preds = knn.predict(X_valid)

    # getting the mean accuracy to get the optimum k value
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_valid, y_valid)

    # since we are using mean accuracy to measure the model performance
    return train_accuracy, test_accuracy


# test function here
def inference_stage(train_set, test_set, k_neighbours):
    '''

    :param train_set:
    :param test_set:
    :param k_neighbours:
    :return:
    '''
    X_train = train_set.drop(config.OUTPUT_FEATURE, axis=1,
                             inplace=False)
    y_train = train_set[config.OUTPUT_FEATURE]

    X_test = test_set.drop(config.OUTPUT_FEATURE, axis=1,
                           inplace=False)
    y_test = test_set[config.OUTPUT_FEATURE]

    # fit the model
    knn = KNeighborsClassifier(n_neighbors=k_neighbours,
                               metric=config.DISTANCE_METRIC)
    knn.fit(X_train, y_train)

    # returning the mean accuracy score
    return knn.score(X_test, y_test)



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
        clean_train_set = fr_obj.cleaning_data(raw_train_set)
        clean_test_set = fr_obj.cleaning_data(raw_test_set)

        # dump the clean datasets
        dl_obj.dump_file(config.CLEAN_TRAIN_DATASET, clean_train_set)
        dl_obj.dump_file(config.CLEAN_TEST_DATASET, clean_test_set)

    elif args.train == 'skfold':

        # load the train dataset
        train_set = dl_obj.load_file(config.CLEAN_TRAIN_DATASET)

        # the train set changing to a tuple hence doing this
        train_set = train_set[0]

        # print(type(train_set))
        # print(train_set.shape)

        skfold_obj = create_folds.SKFold()
        train_set[config.KFOLD_COLUMN_NAME] = -1
        train_set = skfold_obj.create_folds(train_set, config.NUM_FOLDS)
        # we need to maintain the f1 score on average
        # for all 5 folds for all k neighbours
        train_accuracy_avg = []
        valid_accuracy_avg = []

        for k in range(1,51):
            train_accuracy_per_k = []
            valid_accuracy_per_k = []
            for fold_value in range(config.NUM_FOLDS):
                train_accuracy, test_accuracy = run(dataset=train_set,
                                    k_neighbours=k, fold=fold_value)

                train_accuracy_per_k.append(train_accuracy)
                valid_accuracy_per_k.append(test_accuracy)

            # storing the average test score of 5 runs for each k
            train_accuracy_avg.append(np.mean(train_accuracy_per_k))
            valid_accuracy_avg.append(np.mean(valid_accuracy_per_k))

        # print(train_accuracy_avg, valid_accuracy_avg)
        # dump the metric score, as we will use this as a plot
        dl_obj.dump_file(config.TRAIN_ACCURACY_SCORE, train_accuracy_avg)
        dl_obj.dump_file(config.VALIDATION_ACCURACY_SCORE, valid_accuracy_avg)

        print("Model training complete.\n"
              " Please refer to accuracy scores to identify the optimum k value.")

    elif args.test == 'inference':

        # since knn is a lazy learner, we have to fit again with
        # training set before prediction starts

        # load train and test set
        train_set = dl_obj.load_file(config.CLEAN_TRAIN_DATASET)
        test_set = dl_obj.load_file(config.CLEAN_TEST_DATASET)
        
        num_neighbours = 18
        # call the inference stage function and get accuracy score
        test_accuracy = inference_stage(train_set[0], test_set[0],
                                        num_neighbours)
        print(test_accuracy)
