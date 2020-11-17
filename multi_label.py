import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report, recall_score

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_multilabel_classification
from collections import Counter

import pandas as pd
import numpy as np


class MultiLabelEvaluator():
    def __init__(self, model ,X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.confusion_matrices = None
        self.auc_scores = None
        self.accuracies = None
        self.predictions = None


    def make_predictions(self):
        self.predictions = self.model.predict(self.X_test)

    def evaluate_model(self, print_info = False):
        confusion_matrices = {}
        auc_scores = {}
        accuracies = {}

        if print_info:
            print('------------------------------------------')

        for i,col in enumerate(self.y_test.columns):
            if print_info:
                print(f'Column {col}: ')

            cmat = confusion_matrix(self.y_test.iloc[:,i], self.predictions[:,i])
            try:
                auc = roc_auc_score(self.y_test.iloc[:,i], self.predictions[:,i])
            except:
                auc = -1

            acc = accuracy_score(self.y_test.iloc[:,i], self.predictions[:,i])

            if print_info:
                fail_info = 'unable to calculate auc'
                print(cmat)
                print(f'Auc score: {auc if auc >0 else fail_info }')

            confusion_matrices[col] = cmat
            auc_scores[col] = auc
            accuracies[col] = acc

        self.confusion_matrices = confusion_matrices
        self.auc_scores = auc_scores
        self.accuracies = accuracies
        return self.confusion_matrices, self.auc_scores, self. accuracies

    def plot_confusion_matrices(self,savefig = False, n_cols = 5):
        n_charts = len(self.confusion_matrices)
        n_rows = n_charts // n_cols + int(n_charts % n_cols >0)
        fig,ax = plt.subplots(figsize = (30,20))
        for i, (k,v) in enumerate(self.confusion_matrices.items()):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.heatmap(v.astype(int),center = 500,vmax = 50000,fmt = 'd', annot = True)
            plt.title(k + f' auc = {round(self.auc_scores[k],2)}')
        plt.tight_layout()
        if savefig:
            plt.savefig('confusion_matrices.png', dpi = 300)
        plt.show()


    def evaluate(self):
        self.make_predictions()
        return self.evaluate_model()


class MultilabelPredictor(BaseEstimator, ClassifierMixin):

    '''
        A model combining sampling and predict
        Acts like a pipeline
    '''

    def __init__(self, base_predictor = None, over_sampler = None, under_sampler = None,
                 under_sample_minor_to_major = 0.5, over_sample_minor_to_major = 'auto',
                 fit_undersampler = True, fit_oversampler = True):
        '''
            MultilabelPredictor intends to inherit from scikit-learn predictors.
            With some additional functionalities added.
            1. Perform undersampling on majority class (with appropriate minor_major_ratio)
            2. Perform oversampling on minority class (equal minor - major ratio by default)
            3. Use a base predictor to predict each label
            4. fitted classifiers are stored in CamelMultilabelPredictor.classifiers,
                each element in this list is a scikit-learn predictor

            Parameters:
                base_predictor: scikit-learn classifier
                over_sampler: imblearn oversampler
                under_sampler: imblearn under sampler
                under_sample_minor_to_major: undersampling strategy parameter
                over_sample_minor_to_major: oversampling strategy parameter
        '''

        self.classifiers = []
        self.base_predictor = base_predictor if base_predictor else LogisticRegression()
        self.n_classes = None
        self.n_features = None
        self.fitted = False
        self.under_sample_minor_to_major = under_sample_minor_to_major
        self.over_sample_minor_to_major = over_sample_minor_to_major

        self.fit_oversampler = fit_oversampler
        self.fit_undersampler = fit_undersampler

        self.over_sampler = over_sampler if over_sampler \
            else RandomOverSampler()
        self.under_sampler = under_sampler if under_sampler \
            else RandomUnderSampler()

    def calculate_strategy(self, y, sampler, default_strategy):

        '''
            Calculate sampling strategy:
                A sampling strategy is usually fixed, however, when you have multilabel problems,
                You may want different sampling strategy for each label.
                Suppose you have 100 positive and 900 negative examples. you want to resample to balanced class ratio.
                1. Sample 100 positive examples and 300 negative examples, thus, reduced negative examples
                2. Smote sample the 100 positive up to 300, get 300 positives and 300 negatives
                3. Then you have a pos:neg 1:1 dataset, each class containing 300 examples.

                If setting fixed ratio, imblearn sampler may raise an error due to not enough datapoints.
                This method correctify the default strategy, and reduced the chance of raising an error.

            Parameters:
                y: target binary variable
                sampler: imblearn sampler
                default_strategy: user defined default strategy

            Return: An imblearn sampler with proper sampling strategy

        '''

        if default_strategy == 'auto':
            sampler.sampling_strategy = 'auto'
            return sampler

        minor, major = sorted(dict(Counter(y)).items(), key = lambda x:x[1])
        minor_to_major = (minor[1] / major[1]) + 0.01
        sampler.sampling_strategy = max(minor_to_major, default_strategy)
        return sampler

    def under_sampling(self, X, y):
        '''Perform under sampling on dataset'''

        if not self.fit_undersampler:
            return X,y

        sampler = self.calculate_strategy(y, self.under_sampler, self.under_sample_minor_to_major)
        return sampler.fit_resample(X,y)

    def over_sampling(self, X, y):
        '''Perform  over sampling on dataset'''
        if not self.fit_oversampler:
            return X,y

        sampler = self.calculate_strategy(y, self.over_sampler, self.over_sample_minor_to_major)
        return sampler.fit_resample(X,y)

    def combined_resampling(self, X, y):
        '''Perform oversampling followed by usersampling'''
        x_resampled, y_resampled = self.under_sampling(X,y)
        return self.over_sampling(x_resampled, y_resampled)


    def fit(self,X, y, validation_data = None):
        '''Fit function~ aha'''


        self.n_classes = y.shape[1] if y.ndim >1 else 1
        self.n_features = X.shape[1]

        if validation_data:
            X_val, y_val = validation_data
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.values

        if isinstance(y,pd.DataFrame):
            y = y.values


        for col_id in range(self.n_classes):
            print(f'fitting target: {col_id} >>>>>> ', end = '')
            try:
                y_to_fit = y[:, col_id] if self.n_classes > 1 else y
                x_resampled, y_resampled = self.combined_resampling(X,y_to_fit)
            except Exception as e:
                print('Resample error')
                print(e)
                x_resampled, y_resampled = X, y[:, col_id]

            clf = clone(self.base_predictor)
            clf.fit(x_resampled, y_resampled)
            self.classifiers.append(clf)

            if validation_data:
                self.evaluate_current(X_val, y_val[:,col_id], col_id)

            print('Done.')

        self.fitted = True
        return self

    def predict(self, X):
        assert X.shape[1] == self.n_features
        assert self.fitted

        predictions = []
        for col_id in range(self.n_classes):
            predictions.append(self.classifiers[col_id].predict(X))

        return np.array(predictions).T

    def evaluate_current(self, X, y, col_id):

        pred = self.classifiers[col_id].predict(X)
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, pred)
        recall = recall_score(y, pred)
        cmat = confusion_matrix(y, pred)

        print(f'val acc: {acc} | val auc : {auc} | recall: {recall}')
        print('---------------------------------------')




def compare_models(compare_df):
    '''
        This function make a slope graph, comparing 2 models on auc

        Parameters:
            compare_df: a dataframe with 3 columns, each row represents one label
            First column: metric of model 1
            Second column: metric of model 2
            Third column: difference of 2 models

        Return None,

        Draw a slope graph of the dataframe

    '''
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure
    output_notebook()

    model1_name, model2_name, diff_col = compare_df.columns.tolist()

    p = figure(y_range = (0.8,1),
               title = f'Model Auc Comparision : {model1_name} vs {model2_name}',
               x_axis_label = 'Models',
               y_axis_label = 'Auc values', tools = 'hover, box_select, lasso_select, reset')

    for i, row in compare_df.iterrows():
        color = 'lightcoral' if row['difference'] >0 else 'deepskyblue'

        p.line([1,2],[row[model1_name],row[model2_name]],
               line_width = 3, color = color,)

        p.circle([1,2],[row[model1_name],row[model2_name]], size = 10,fill_color = 'white', line_width = 4,
                 color = color )

    p.xaxis.ticker = [0,1,2,3]
    p.xaxis.major_label_overrides = {1:model1_name,2:model2_name}
    show(p)
