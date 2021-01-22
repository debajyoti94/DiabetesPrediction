''' Here we define all the unit test cases'''

# import modules here
import config
import feature_engg

import pandas as pd
import csv

import os


# test class here
class TestFunctionalities:

    # test: delimiter check
    def test_delimiter_check(self):
        '''
        Check if the file delimiter matches with
         what is provided in config file
        :return: true if all is okay
        '''
        with open(config.ORIGINAL_RAW_DATASET, 'r') as csv_file:
            file_content = csv.Sniffer().sniff(csv_file.readline())

        assert True if file_content.delimiter == config.FILE_DELIMITER \
            else False


    # test: dataset shape check
    def test_dataset_shape(self):
        '''
        check if dataset shape matches with what is expected
        :return: True if all is okay
        '''
        dataset = pd.read_csv(config.ORIGINAL_RAW_DATASET,
                              delimiter=config.FILE_DELIMITER,
                              encoding=config.ENCODING_TYPE)

        assert True if dataset.shape == config.DATASET_SHAPE else False

    # test: file check
    def test_file_check(self):
        '''
        check if the clean test and train exists
        :return:
        '''
        assert True if os.path.isfile(config.CLEAN_TRAIN_DATASET) and\
                       os.path.isfile(config.CLEAN_TEST_DATASET)\
            else False

    # test: null check
    def test_null_check(self):
        '''
        Check if any null value exists after preprocessing
        :return:
        '''
        dl_obj = feature_engg.DumpLoadFile()
        clean_train = dl_obj.load_file(config.CLEAN_TRAIN_DATASET)
        clean_test = dl_obj.load_file(config.CLEAN_TEST_DATASET)

        assert False if False in pd.isnull(clean_train) or pd.isnull(clean_test)\
            else True


