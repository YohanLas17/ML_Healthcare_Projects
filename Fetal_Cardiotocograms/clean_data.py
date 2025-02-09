# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def rm_ext_and_nan(CTG_features, extra_feature):
    """
    Removes the specified feature and rows with NaN values.
    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A cleaned dataframe with NaN values dropped
    """
    CTG_features = CTG_features.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    # Remove the specified feature
    CTG_features.drop([extra_feature], axis=1, inplace=True)
    # Convert to numeric and coerce errors to NaN
    CTG_features = CTG_features.apply(pd.to_numeric, errors='coerce')
    # Drop rows with NaN values
    c_ctg = CTG_features.dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """
    Replaces NaN values with random values from the same column.
    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe with NaN values replaced
    """
    CTG_features = CTG_features.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features.drop([extra_feature], axis=1, inplace=True)
    CTG_features = CTG_features.apply(pd.to_numeric, errors='coerce')

    # Set a seed for reproducibility
    np.random.seed(42)

    # Replace missing values with random values from the same column
    c_cdf = CTG_features.apply(
        lambda col: col.apply(lambda x: np.random.choice(col.dropna().values) if pd.isna(x) else x), axis=0)
    # --------------------------------------------------------------------------
    return c_cdf


def sum_stat(c_feat):
    """
    Computes summary statistics for each column of the cleaned data.
    :param c_feat: Cleaned dataframe
    :return: Dictionary of summary statistics
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for col in c_feat.columns:
        d_summary[col] = {
            "min": c_feat[col].min(),
            "Q1": c_feat[col].quantile(0.25),
            "median": c_feat[col].median(),
            "Q3": c_feat[col].quantile(0.75),
            "max": c_feat[col].max()}
    # --------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """
    Removes outliers from the cleaned data using the IQR method.
    :param c_feat: Cleaned dataframe
    :param d_summary: Summary statistics
    :return: Dataframe with outliers removed
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_no_outlier = c_feat.copy()
    for col in c_no_outlier.columns:
        Q1 = d_summary[col]["Q1"]
        Q3 = d_summary[col]["Q3"]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace values outside the bounds with NaN
        c_no_outlier[col] = c_no_outlier[col].where(
            (c_no_outlier[col] >= lower_bound) & (c_no_outlier[col] <= upper_bound), np.nan)
    # --------------------------------------------------------------------------
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """
    Filters the feature by threshold.
    :param c_samp: Cleaned dataframe
    :param feature: Name of the feature to filter
    :param thresh: Threshold for filtering values
    :return: Filtered feature as a numpy array
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    filt_feature = c_samp[feature].apply(lambda x: np.nan if x > thresh else x)
    # --------------------------------------------------------------------------
    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False

    def fit(self, CTG_features):
        """
        Calculate min, max, mean, and std for each feature.
        :param CTG_features: DataFrame of CTG features
        """
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        self.max = CTG_features.max()
        self.min = CTG_features.min()
        self.mean = CTG_features.mean()
        self.std = CTG_features.std()
        self.fit_called = True
        # --------------------------------------------------------------------------

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Transforms the features based on the selected mode ('none', 'standard', 'MinMax', 'mean').
        :param CTG_features: The features to transform
        :param mode: The transformation mode ('none', 'standard', 'MinMax', 'mean')
        :param selected_feat: Tuple of features to plot
        :param flag: Whether to plot the histogram
        :return: Transformed dataframe
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U.]'
            elif mode == 'standard':
                nsd_res = (ctg_features - self.mean) / self.std
                x_lbl = 'Standardized values [N.U.]'
            elif mode == 'MinMax':
                nsd_res = (ctg_features - self.min) / (self.max - self.min)
                x_lbl = 'Normalized values [N.U.]'
            elif mode == 'mean':
                nsd_res = ctg_features - self.mean
                x_lbl = 'Mean normalized values [N.U.]'
            else:
                raise ValueError("Invalid mode. Choose 'none', 'standard', 'MinMax', or 'mean'.")

            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            # --------------------------------------------------------------------------
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Fits the model to the data and transforms it.
        :param CTG_features: The features to transform
        :param mode: The transformation mode ('none', 'standard', 'MinMax', 'mean')
        :param selected_feat: Tuple of features to plot
        :param flag: Whether to plot the histogram
        :return: Transformed dataframe
        """
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        """
        Plots histograms for two selected features.
        :param nsd_res: The normalized or standardized dataframe
        :param mode: The mode of transformation
        :param selected_feat: The two features to plot
        :param x_lbl: Label for the x-axis
        """
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80

        # Create the plot
        plt.figure(figsize=(10, 5))

        # Histogram for the first feature (x)
        plt.hist(nsd_res[x], bins=bins, alpha=0.4, label=f'{x} - {x_lbl}', color='blue')

        # Histogram for the second feature (y)
        plt.hist(nsd_res[y], bins=bins, alpha=0.4, label=f'{y} - {x_lbl}', color='red')

        # Add titles and labels
        plt.title(f'Histograms for {mode} mode')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Add legend
        plt.legend(loc='upper right')

        # Display the plot
        plt.show()


# Debugging block!
if __name__ == '__main__':
    file = Path.cwd().joinpath('messed_CTG.xls')
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    extra_feature = 'DR'

    # Clean the data by removing the extra feature and NaN values
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)

    # Replace NaN values with random samp
