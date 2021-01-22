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
class FeatureEngineering(MustHaveForFeatureEngineering):

    def scale_features(self, features):
        '''
        Applying standard scaling here
        :param features: features to be scaled
        :return: scaled features
        '''
        features_scaled = preproc.StandardScaler.fit_transform(features)

        return features_scaled


    def cleaning_data(self, dataset):
        '''
        Provide cleaned dataset
        :param dataset: input dataset to be cleaned
        :return: cleaned dataset
        '''

        # apply feature scaling here
        features = dataset.drop(config.OUTPUT_FEATURE, axis=1, inplace=False)
        target = dataset[config.OUTPUT_FEATURE]

        features_scaled = self.scale_features(features)
        features_scaled[config.OUTPUT_FEATURE]  = target

        return features_scaled

    def null_plot(self, dataset):
        """
                Create a heatmap to verify if any null value exists
                :param dataset: dataset to be verified
                :return: heatmap plot
        """
        sns_heatmap_plot = sns.heatmap(
            dataset.isnull(), cmap="Blues", yticklabels=False
        )
        sns_heatmap_plot.figure.savefig(config.NULL_CHECK_HEATMAP)


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