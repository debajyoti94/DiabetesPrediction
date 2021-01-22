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

    parser = argparse.ArgumentParser()

    # adding the arguments

#commandline arguments here