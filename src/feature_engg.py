''' Here we will apply the feature engineering techniques on the dataset'''


# import modules here
import config
import pickle

from sklearn import preprocessing as preproc
import seaborn as sns
import abc

# abc class here
class MustHaveForFeatureEngineering:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cleaning_data(self):
        return

    @abc.abstractmethod
    def null_plot(self):
        return

# child class here


# dump and load object class here
class DumpLoadFile:

    def load_file(self, filename):
        '''
        For loading the pickled files
        :param filename: the file that you want to load
        :return: the loaded file
        '''
        with open(filename, 'rb') as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, file, filename):
        '''
        Pickle a file
        :param file: file that you want to pickle
        :param filename: filename for the pickled file
        :return: nothing
        '''
        with open(filename, 'wb') as pickle_handle:
            pickle.dump(file, pickle_handle)