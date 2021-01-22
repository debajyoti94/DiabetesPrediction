''' Here we write the code for applying stratified k fold cross validation'''

from sklearn import model_selection
import config


class SKFold:

    def create_folds(self, num_folds, dataset):
        '''
        In this code we will assign values to the column kfold
        and use that during training the model
        :param num_folds:
        :return: dataset with kfold value
        '''

        kf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True,
                                             random_state=0)

        y = dataset[config.OUTPUT_FEATURE].values

        for fold_value, (t_, y_index) in enumerate(kf.split(X=dataset, y=y)):
            dataset.loc[y_index, config.KFOLD_COLUMN_NAME] = fold_value

        return dataset