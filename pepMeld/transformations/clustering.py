import pandas as pd
from .utils import *

'''
## Physical Array Spatial Clustering 
- Identifies smears in the array
- Identifies hotspots in array
- First does a 45 degree /2 transformation on the X-Y array because the array is spaced like a checker board
- Next Loops through each data column matrix
- Adds filter for x %-tile gaps
- NxN area  3x3 up to 11 x 11 Squares for now will advance to plygon inside a circle later.
'''


def plot_cluster_map(X,
                     quantile_group,
                     range_low,
                     range_high,
                     range_axis,
                     data_column_i,
                     OUTPUT_CHARTS_DIR,
                     number_to_name_dict={},
                     eps=4,
                     min_samples=10,
                     save_plot=True):

    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    import os
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    n_clusters_points_ = len(X) - n_noise_

    if n_clusters_ > 0:
        print('Data Column Name: %s' % data_column_i)
        print('Quantile Group: %s' % quantile_group)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of clusters points: %d' % n_clusters_points_)
        print('Estimated number of noise points: %d' % n_noise_)
        print('Range between : %s and %s' % (range_low, range_high))
        # #############################################################################
        # Plot result
        if save_plot:
            import matplotlib.pyplot as plt

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], '.', color=tuple(col), markersize=3)

            # xy = X[class_member_mask & ~core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #		 markeredgecolor='k', markersize=1)

            plt.title('Between %s and %s : %d clusters and % d points' % (
                round(range_low, 3), round(range_high, 3), n_clusters_, n_clusters_points_))
            plt.axis(range_axis)

            F = plt.gcf()
            chart_name = data_column_i
            if bool(number_to_name_dict):
                if str(data_column_i) in number_to_name_dict:
                    chart_name = number_to_name_dict[str(data_column_i)]


            F.savefig(os.path.join(OUTPUT_CHARTS_DIR, '{0}_quantile_group_{1}.png'.format(chart_name, quantile_group)),
                      dpi=(500))
            # plt.show()
            plt.clf()
    return (labels)


'''
multipthreaded cluster map
'''


def plot_cluster_map_multi(df_filtered_dict,
                           # quantile_group,

                           range_axis,
                           data_column_i,
                           OUTPUT_CHARTS_DIR,
                           number_to_name_dict={},
                           eps=4,
                           min_samples=10,
                           save_plot=True):
    # print('start_cluster_map')
    import numpy as np
    import pandas as pd
    import os
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    quantile_group = df_filtered_dict['quantile_group']
    range_low = df_filtered_dict['range_low']
    range_high = df_filtered_dict['range_high']
    df_filtered = df_filtered_dict['df_filtered']
    X = df_filtered[['X', 'Y']].values

    # #############################################################################
    # Compute DBSCAN
    # print(range_low)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    # print(range_high)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    n_clusters_points_ = len(X) - n_noise_
    if n_clusters_ > 0:
        print('Data Column Name: %s' % data_column_i)
        print('Quantile Group: %s' % quantile_group)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of clusters points: %d' % n_clusters_points_)
        print('Estimated number of noise points: %d' % n_noise_)
        print('Range between : %s and %s' % (range_low, range_high))
        # #############################################################################
        # Plot result
        if save_plot:
            chart_name = data_column_i
            print('chartname {0}'.format(chart_name))
            if bool(number_to_name_dict):
                print('dict exists')
                if str(data_column_i) in number_to_name_dict:
                    print('keyexists')
                    chart_name = number_to_name_dict[str(data_column_i)]
            print('chartname after : {0}'.format(chart_name))
            import matplotlib.pyplot as plt

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], '.', color=tuple(col), markersize=3)

            # xy = X[class_member_mask & ~core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #		 markeredgecolor='k', markersize=1)

            plt.title('Between %s and %s : %d clusters and % d points' % (
                round(range_low, 3), round(range_high, 3), n_clusters_, n_clusters_points_))
            plt.axis(range_axis)

            F = plt.gcf()

            F.savefig(os.path.join(OUTPUT_CHARTS_DIR, '{0}_quantile_group_{1}.png'.format(chart_name, quantile_group)),
                      dpi=(500))
            plt.show()
            plt.clf()
    df_filtered['CLUSTER'] = labels
    df_filtered = df_filtered.loc[df_filtered['CLUSTER'] != -1]
    df_filtered['CLUSTER_COUNT'] = df_filtered.groupby('CLUSTER')['CLUSTER'].transform('count')

    df_filtered['PERCENTILE'] = [range_low] * len(df_filtered.index)
    return df_filtered


# return (labels)

'''
Clustering algorithm, multithreaded
'''


def cluster_multi(quantile_group,  # contant_dict):
                  data_column_i,  # = contant_dict['data_column_i']
                  range_axis,  # = contant_dict[ 'range_axis']
                  percentile,  # = contant_dict['percentile']
                  OUTPUT_CHARTS_DIR,  # = contant_dict['OUTPUT_CHARTS_DIR']
                  eps,  # = contant_dict['eps']
                  min_samples,  # = contant_dict['min_samples']
                  df_i,
                  number_to_name_dict={}):
    df_filtered = df_i.loc[
        (df_i[data_column_i] >= percentile[quantile_group]) & (df_i[data_column_i] <= percentile[quantile_group + 2])][
        ['X', 'Y']]
    X = df_filtered[['X', 'Y']].values
    labels = plot_cluster_map(X, quantile_group,
                              percentile[quantile_group],
                              percentile[quantile_group + 2],
                              range_axis,
                              data_column_i,
                              OUTPUT_CHARTS_DIR,
                              number_to_name_dict=number_to_name_dict,
                              eps=eps,
                              min_samples=min_samples,
                              save_plot=False)
    df_filtered['CLUSTER'] = labels
    df_filtered = df_filtered.loc[df_filtered['CLUSTER'] != -1]
    df_filtered['CLUSTER_COUNT'] = df_filtered.groupby('CLUSTER')['CLUSTER'].transform('count')

    df_filtered['PERCENTILE'] = [percentile[quantile_group]] * len(df_filtered.index)
    return df_filtered


'''
convex hull
'''


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
        in_hull_array = hull.find_simplex(p) >= 0
    return p[in_hull_array]


class clustering:
    def expand_cluster_convexhull(self, df_in_name, df_cluster_name, df_cluster_expanded_name):
        from scipy.spatial import ConvexHull
        if len(self.df_dict[df_cluster_name]) < 1:
            print('NO CLUSTERS EXIST:')
            print(df_cluster_expanded_name)
            self.df_dict[df_cluster_expanded_name] = pd.DataFrame(columns=['CLUSTER',
                                                                           self.sample_name_column,
                                                                           'PERCENTILE',
                                                                           'CLUSTER_COUNT',
                                                                           'ORIGINAL_CLUSTER_COUNT',
                                                                           'ORIGINAL_CLUSTER',
                                                                           'CLUSTER_RATIO',
                                                                           'CLUSTERED',
                                                                           'EXPANDED'])
            return
        print('CLUSTERS EXIST:')
        print(df_cluster_expanded_name)
        cluster_names_df = self.df_dict[df_cluster_name][
            ['CLUSTER', self.sample_name_column, 'PERCENTILE', 'CLUSTER_COUNT']].drop_duplicates(subset=None,
                                                                                                 keep='first',
                                                                                                 inplace=False)
        self.df_dict[df_cluster_expanded_name] = pd.DataFrame()
        for index, row in cluster_names_df.iterrows():
            # get the cluster
            cluster_points_df_i = self.df_dict[df_cluster_name].loc[
                (self.df_dict[df_cluster_name]['PERCENTILE'] == row['PERCENTILE']) & (
                        self.df_dict[df_cluster_name]['CLUSTER'] == row['CLUSTER']) & (
                        self.df_dict[df_cluster_name][self.sample_name_column] == row[self.sample_name_column])][
                ['X', 'Y']]
            # convert DF to an array
            cluster_points_array = cluster_points_df_i.values
            # get the hull for the cluster
            hull = ConvexHull(cluster_points_array)
            # get the vertices of the hull
            hull_vertices = cluster_points_array[hull.vertices]
            # get points inside the hull vertices as a numpy array
            contained_points = in_hull(self.df_dict[df_in_name][['X', 'Y']].values, hull_vertices)
            # contained_points
            # convert back to dataframe
            contained_points_df = pd.DataFrame(contained_points, columns=['X', 'Y'])
            # name the cluster, Sample name, percentile and oritinal clustercount
            contained_points_df['CLUSTER'] = [row['CLUSTER']] * len(contained_points_df.index)
            contained_points_df[self.sample_name_column] = [row[self.sample_name_column]] * len(
                contained_points_df.index)
            contained_points_df['PERCENTILE'] = [row['PERCENTILE']] * len(contained_points_df.index)
            contained_points_df['ORIGINAL_CLUSTER_COUNT'] = [row['CLUSTER_COUNT']] * len(contained_points_df.index)
            # get the newly expanded cluster count
            contained_points_df['CLUSTER_COUNT'] = contained_points_df.groupby('CLUSTER')['CLUSTER'].transform('count')
            # Label the points that are part of the original cluster
            # get the points of the original cluster
            original_cluster_points_df_i = cluster_points_df_i[['X', 'Y']]
            # label this data frame as the original
            original_cluster_points_df_i['ORIGINAL_CLUSTER'] = ['ORIGINAL'] * len(original_cluster_points_df_i.index)
            contained_points_df = contained_points_df.merge(original_cluster_points_df_i, how='outer', on=['X', 'Y'])
            # contained_points_df
            self.df_dict[df_cluster_expanded_name] = pd.concat(
                [self.df_dict[df_cluster_expanded_name], contained_points_df], ignore_index=True)
        self.df_dict[df_cluster_expanded_name].ORIGINAL_CLUSTER.fillna('EXPANDED', inplace=True)
        self.df_dict[df_cluster_expanded_name]['CLUSTERED'] = ['CLUSTERED'] * len(
            self.df_dict[df_cluster_expanded_name].index)
        self.df_dict[df_cluster_expanded_name]['CLUSTER_RATIO'] = round(
            self.df_dict[df_cluster_expanded_name]['ORIGINAL_CLUSTER_COUNT'] / self.df_dict[df_cluster_expanded_name][
                'CLUSTER_COUNT'], 2)

    # Finds clusters and expanded clusters

    # df_drop_list = list(set(lower_df.index).intersection(df_filtered.index))
    # df_filtered.drop(df_drop_list, inplace = True)
    # df_filtered.drop(columns = ['X','Y'], inplace = True)
    # return df_filtered
    def find_clusters(self, transformation_class):
        import os

        df_in_name = transformation_class.df_in_name
        df_cluster_name = transformation_class.df_cluster_name
        df_out_name = transformation_class.df_out_name
        OUTPUT_CHARTS_DIR = transformation_class.OUTPUT_CHARTS_DIR
        merge_to_input_df = transformation_class.merge_to_input_df
        percentile_slices = transformation_class.percentile_slices
        eps = transformation_class.eps
        min_samples = transformation_class.min_samples
        save_plot = transformation_class.save_plot

        if OUTPUT_CHARTS_DIR is None:
            if save_plot:
                print('OUTPUT_CHARTS_DIR to save the plots to, changing save_plot to False')
                save_plot = False

        if ("X" not in list(self.df_dict[df_in_name].columns)) and ("Y" not in list(self.df_dict[df_in_name].columns)):
            print("missing columns X and/or Y.  Cannot complete clustering")
            self.df_dict[df_cluster_name] = pd.DataFrame()
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy()
            return
        # make sure the slices are appropriate if not print a warning message
        if percentile_slices < 10:
            print("INFORMATION: Slices are not very narrow, you may want more slices for more percise clustering")
        # cahnge to the correct nomenclature
        percentile_slices = percentile_slices
        percentile_slices_used = percentile_slices * 2
        # get quantiles for double the slices so you can have 50% overlapping in the regions
        quantiles = [x / (percentile_slices_used - 1) for x in range(percentile_slices_used)]
        # make an empty data frame the will be concatenated on as a loop.
        self.df_dict['CLUSTER_TEMP'] = pd.DataFrame()
        print(self.number_to_name_dict)
        # data_column_i = 'BLANK_188996_IgG_635_A12.dat'
        for data_column_i in self.data_columns:
            # data_column_i = '70'
            print('Data_Column: {0}'.format(data_column_i))
            df = self.df_dict[df_in_name][['X', 'Y', data_column_i]]
            percentile = list(df[data_column_i].quantile(quantiles))
            range_axis = [df['X'].min(), df['X'].max(), df['Y'].min(), df['Y'].max()]
            lower_df = pd.DataFrame()

            # cpu_count =  os.cpu_count()
            single_cpu = False
            if single_cpu:
                for quantile_group in range(0, len(percentile) - 2, 1):
                    df_filtered = df.loc[(df[data_column_i] >= percentile[quantile_group]) & (
                            df[data_column_i] <= percentile[quantile_group + 2])][['X', 'Y']]
                    X = df_filtered[['X', 'Y']].values
                    labels = plot_cluster_map(X, quantile_group,
                                              percentile[quantile_group],
                                              percentile[quantile_group + 2],
                                              range_axis,
                                              data_column_i,
                                              OUTPUT_CHARTS_DIR,
                                              number_to_name_dict=self.number_to_name_dict,
                                              eps=eps,
                                              min_samples=min_samples,
                                              save_plot=False)
                    df_filtered['CLUSTER'] = labels
                    df_filtered = df_filtered.loc[df_filtered['CLUSTER'] != -1]
                    df_filtered['CLUSTER_COUNT'] = df_filtered.groupby('CLUSTER')['CLUSTER'].transform('count')

                    df_filtered['PERCENTILE'] = [percentile[quantile_group]] * len(df_filtered.index)

                    df_drop_list = list(set(lower_df.index).intersection(df_filtered.index))
                    df_filtered.drop(df_drop_list, inplace=True)
                    # df_filtered.drop(columns = ['X','Y'], inplace = True)
                    lower_df = pd.concat([lower_df, df_filtered], ignore_index=False)
            else:

                from functools import partial

                cpu_count = os.cpu_count()

                # percentile_slices = 20
                # percentile_slices_used = percentile_slices * 2
                # get quantiles for double the slices so you can have 50% overlapping in the regions
                # quantiles = [x / (percentile_slices_used-1) for x in range(percentile_slices_used)]
                # df_i = df#[['X','Y',data_column_i]]
                # percentile = list(df[data_column_i].quantile(quantiles))
                # range_axis = [df['X'].min(),df['X'].max(),df['Y'].min(),df['Y'].max()]
                # lower_df = pd.DataFrame()
                max_count = len(percentile) - 2

                # contant_dict = {'data_column_i' : data_column_i,
                #				  'range_axis' : range_axis,
                #				  'percentile' : percentile,
                #				  'OUTPUT_CHARTS_DIR' : OUTPUT_CHARTS_DIR,
                #				 'eps' : 3,
                #				 'min_samples':10}
                # df_filtered = df_i.loc[(df_i[data_column_i] >= percentile[quantile_group]) & (df_i[data_column_i] <=  percentile[quantile_group+2])][['X','Y']]

                plot_cluster_map_multi_partial = partial(plot_cluster_map_multi,
                                                         range_axis=range_axis,
                                                         data_column_i=data_column_i,
                                                         OUTPUT_CHARTS_DIR=OUTPUT_CHARTS_DIR,
                                                         number_to_name_dict=self.number_to_name_dict,
                                                         eps=eps,
                                                         min_samples=min_samples,
                                                         save_plot=False)

                # lower_df = pd.DataFrame()
                from contextlib import closing
                for lower_range in range(0, max_count, cpu_count):
                    import multiprocessing as mp
                    df_filtered_dict_list = []
                    print(lower_range)
                    # print(df_filtered_tupple_list)
                    upper_range = lower_range + cpu_count
                    if upper_range > max_count:
                        upper_range = max_count
                    for quantile_group in range(lower_range, upper_range, 1):
                        df_filtered = df.loc[(df[data_column_i] >= percentile[quantile_group]) & (
                                df[data_column_i] <= percentile[quantile_group + 2])][['X', 'Y']]
                        df_filtered_dict_list.append({'quantile_group': quantile_group,
                                                      'range_low': percentile[quantile_group],
                                                      'range_high': percentile[quantile_group + 2],
                                                      'df_filtered': df_filtered})

                    with closing(mp.Pool(cpu_count)) as pool:
                        # pool = mp.Pool(processes=cpu_count)
                        df_array = pool.map(plot_cluster_map_multi_partial, df_filtered_dict_list)
                    df_array = pd.concat(df_array, ignore_index=True)
                    lower_df = pd.concat([lower_df, df_array], ignore_index=True)

            if len(lower_df.index) > 0:

                df_lower_perc = lower_df.loc[lower_df['PERCENTILE'] < percentile[len(percentile) - 5]]
                df_lower_perc.drop(columns=['PERCENTILE'], inplace=True)
                if len(df_lower_perc.index) > 0:
                    X = df_lower_perc[['X', 'Y']].values
                    labels = plot_cluster_map(X,
                                              'low',
                                              percentile[0],
                                              percentile[len(percentile) - 4],
                                              range_axis,
                                              data_column_i,
                                              OUTPUT_CHARTS_DIR,
                                              eps=eps,
                                              min_samples=min_samples,
                                              save_plot=save_plot)

                    df_lower_perc['CLUSTER'] = labels
                    df_lower_perc = df_lower_perc.loc[df_lower_perc['CLUSTER'] != -1]
                    df_lower_perc['CLUSTER_COUNT'] = df_lower_perc.groupby('CLUSTER')['CLUSTER'].transform('count')
                    df_lower_perc['PERCENTILE'] = ['low'] * len(df_lower_perc.index)

                df_high_perc = lower_df.loc[lower_df['PERCENTILE'] > percentile[len(percentile) - 5]]
                df_high_perc.drop(columns=['PERCENTILE'], inplace=True)
                if len(df_high_perc.index) > 0:
                    X = df_high_perc[['X', 'Y']].values
                    labels = plot_cluster_map(X,
                                              'high',
                                              percentile[len(percentile) - 4],
                                              percentile[len(percentile) - 1],
                                              range_axis,
                                              data_column_i,
                                              OUTPUT_CHARTS_DIR,
                                              number_to_name_dict=self.number_to_name_dict,
                                              eps=eps,
                                              min_samples=min_samples)
                    df_high_perc['CLUSTER'] = labels
                    df_high_perc = df_high_perc.loc[df_high_perc['CLUSTER'] != -1]
                    df_high_perc['CLUSTER_COUNT'] = df_high_perc.groupby('CLUSTER')['CLUSTER'].transform('count')
                    df_high_perc['PERCENTILE'] = ['high'] * len(df_high_perc.index)
                lower_df = pd.concat([df_lower_perc, df_high_perc], ignore_index=False)

                lower_df[self.sample_name_column] = [data_column_i] * len(lower_df.index)
                self.df_dict['CLUSTER_TEMP'] = pd.concat([self.df_dict['CLUSTER_TEMP'], lower_df], ignore_index=True)
        self.expand_cluster_convexhull(df_in_name, df_cluster_name='CLUSTER_TEMP',
                                       df_cluster_expanded_name=df_cluster_name)
        self.remove_dataframe('CLUSTER_TEMP')
        # merge the file and make sure the additional columns from the  DF_CLUSTER make it into the descriptor columns
        if merge_to_input_df:
            transformation_class = melt_class(df_in_name=df_in_name,
                                              df_out_name=df_out_name,
                                              value_vars=[],
                                              id_vars=[],
                                              value_name='INTENSITY',
                                              var_name=self.sample_name_column)
            self.melt_df(transformation_class)

            transformation_class = merge_data_frames(df_name_left=df_out_name,
                                                     df_name_right=df_cluster_name,
                                                     df_out=df_out_name,
                                                     on_columns=[],
                                                     how='outer')
            self.merge_data_frames(transformation_class)
            self.df_dict[df_out_name].CLUSTER.fillna(-1, inplace=True)
            self.df_dict[df_out_name].PERCENTILE.fillna('NONE', inplace=True)
            self.df_dict[df_out_name].ORIGINAL_CLUSTER_COUNT.fillna(0, inplace=True)
            self.df_dict[df_out_name].CLUSTER_COUNT.fillna(-1, inplace=True)
            self.df_dict[df_out_name].ORIGINAL_CLUSTER.fillna('NONE', inplace=True)
            self.df_dict[df_out_name].CLUSTERED.fillna('NONE', inplace=True)
            self.df_dict[df_out_name].CLUSTER_RATIO.fillna(0, inplace=True)
            self.df_dict[df_out_name]['EXCLUDE'] = self.df_dict[df_out_name]['CLUSTER'].apply(
                lambda x: True if x > -1 else False)
            print(self.df_dict[df_out_name].columns)

    def exclude_clustered_data(self, transformation_class):
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        descriptor_columns = transformation_class.descriptor_columns
        unstack = transformation_class.unstack
        filter_clusters = transformation_class.filter_clusters
        filter_expanded = transformation_class.filter_expanded
        min_original_cluster_size = transformation_class.min_original_cluster_size
        min_cluster_ratio = transformation_class.min_cluster_ratio
        sample_name_col = transformation_class.sample_name_col
        intensity_col = transformation_class.intensity_col

        if 'ORIGINAL_CLUSTER' not in list(self.df_dict[df_in_name].columns):
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy()
            print('missing clustering step, returning input data frame')
            return
        if sample_name_col is None:
            sample_name_col = self.sample_name_column

        if not descriptor_columns:
            descriptor_columns = list(set(self.descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))
        else:
            descriptor_columns = list(set(descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))
        self.df_dict[df_out_name] = self.df_dict[df_in_name]
        if filter_clusters:
            self.df_dict[df_out_name] = self.df_dict[df_out_name].loc[~(
                    (self.df_dict[df_out_name]['ORIGINAL_CLUSTER'] == 'ORIGINAL') & (
                    self.df_dict[df_out_name]['ORIGINAL_CLUSTER_COUNT'] >= min_original_cluster_size) & (
                            self.df_dict[df_out_name]['CLUSTER_RATIO'] >= min_cluster_ratio))]
        if filter_expanded:
            self.df_dict[df_out_name] = self.df_dict[df_out_name].loc[~(
                    (self.df_dict[df_out_name]['ORIGINAL_CLUSTER'] == 'EXPANDED') & (
                    self.df_dict[df_out_name]['ORIGINAL_CLUSTER_COUNT'] >= min_original_cluster_size) & (
                            self.df_dict[df_out_name]['CLUSTER_RATIO'] >= min_cluster_ratio))]
        if unstack:
            self.df_dict[df_out_name] = self.df_dict[df_out_name].pivot_table(index=descriptor_columns,
                                                                              columns=sample_name_col,
                                                                              values=intensity_col).reset_index()
        # self.descriptor_columns = set(self.df_dict[df_out_name].columns) - set(self.data_column_class)
        return


class find_clusters:
    def __init__(self,
                 df_in_name='df',
                 df_cluster_name='expanded_cluster',
                 df_out_name='df_clustered',
                 OUTPUT_CHARTS_DIR=None,
                 merge_to_input_df=True,
                 percentile_slices=20,
                 eps=4,
                 min_samples=10,
                 save_plot=True):
        self.transformation_name = 'find_clusters'
        self.df_in_name = df_in_name
        self.df_cluster_name = df_cluster_name
        self.df_out_name = df_out_name
        self.OUTPUT_CHARTS_DIR = OUTPUT_CHARTS_DIR
        self.merge_to_input_df = merge_to_input_df
        self.percentile_slices = percentile_slices
        self.eps = eps
        self.min_samples = min_samples
        self.save_plot = save_plot
        if OUTPUT_CHARTS_DIR is None:
            if save_plot:
                print('OUTPUT_CHARTS_DIR to save the plots to, changing save_plot to False')
                self.save_plot = False


class exclude_clustered_data:
    def __init__(self,
                 df_in_name='df_clustered',
                 df_out_name=None,
                 descriptor_columns=[],
                 unstack=True,
                 filter_clusters=True,
                 filter_expanded=True,
                 min_original_cluster_size=10,
                 min_cluster_ratio=0,
                 sample_name_col=None,
                 intensity_col='INTENSITY'):
        self.transformation_name = 'exclude_clustered_data'
        self.df_in_name = df_in_name
        self.df_out_name = df_out_name
        self.descriptor_columns = descriptor_columns
        self.unstack = unstack
        self.filter_clusters = filter_clusters
        self.filter_expanded = filter_expanded
        self.min_original_cluster_size = min_original_cluster_size
        self.min_cluster_ratio = min_cluster_ratio
        self.sample_name_col = sample_name_col
        self.intensity_col = intensity_col
