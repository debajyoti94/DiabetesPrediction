'''Here we will train and test the model'''

# import modules here
import config
import feature_engg
import create_folds

import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# train function here
def run(dataset, fold):
    '''

    :param dataset:
    :param fold:
    :return:
    '''
    return

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
        clean_train_set = fr_obj.clean_dataset(raw_train_set)
        clean_test_set = fr_obj.clean_dataset(raw_test_set)

        # dump the clean datasets
        dl_obj.dump_file(clean_train_set, config.CLEAN_TRAIN_DATASET)
        dl_obj.dump_file(clean_train_set, config.CLEAN_TEST_DATASET)

    elif args.train == 'skfold':
        pass

    elif args.test == 'inference':
        pass