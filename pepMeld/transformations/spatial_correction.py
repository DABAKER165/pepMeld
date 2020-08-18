'''
Compute the lowess part and the plots.
'''


def compute_lowess(df=None,
                   frac=0.3,
                   it=3,
                   delta_factor=0.01,
                   y_col='INTENSITY',
                   x_col='SURROUND_MEAN',
                   y_pred_col='PRED_INTENSITY'
                   ):
    # %matplotlib inline
    from scipy.interpolate import interp1d
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np

    df_na = df.loc[np.isnan(df[y_col])]
    df = df.loc[~np.isnan(df[y_col])]

    # Calculate the delta for a 1000x speed boost, other wise it will take hours.
    delta = delta_factor * (max(df[x_col]) - min(df[x_col]))
    print("delta: {0}".format(delta))
    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess2 = sm.nonparametric.lowess(df[y_col], df[x_col], frac=frac, it=it, delta=delta)
    df2 = pd.DataFrame(lowess2, columns=[x_col, y_pred_col])
    df2.drop_duplicates(inplace=True)
    df = df.merge(df2, how='inner', on=[x_col])

    df = pd.concat([df,df_na], ignore_index = True)

    # return on the (panda) list of data
    # print(df2)
    return (df)


def plot_lowess(df,
                y_col,
                x_col,
                y_pred_col,
                OUTPUT_CHARTS_DIR,
                chart_subname):
    import matplotlib.pyplot as plt
    from os import path
    plt.plot(df[x_col], df[y_col], ',')
    plt.plot(df[x_col], df[y_pred_col], '.')

    F = plt.gcf()
    F.savefig(path.join(OUTPUT_CHARTS_DIR, '{0}_lowess_plot.png'.format(chart_subname)), dpi=(500))
    plt.show()
    plt.clf
    # plt
    return;


'''
Compute Multi-Threaded Lowess.  
'''


def compute_multi_lowess(df_spatial_temp, y_col, x_col, y_pred_col, empty_slot_value,
                         OUTPUT_CHARTS_DIR, frac=0.3, it=3, delta_factor=0.01, save_plot=True, save_table=False, number_to_name_dict={}):
    import numpy as np
    import os
    import pandas as pd
    # import multiprocessing as mp
    # import os
    # from functools import partial
    # print(constant_dict)
    # frac =constant_dict['frac']
    # it =constant_dict['it']
    # delta_factor=constant_dict['delta_factor']
    # y_col =constant_dict['y_col']
    # x_col = constant_dict['x_col']
    # y_pred_col =constant_dict['y_pred_col']
    shift_array = [[0, 2], [0, -2], [1, 1], [-1, 1], [1, -1], [-1, -1]]
    i = 0

    sample_number = list(set(df_spatial_temp.columns) - {'X', 'Y'})[0]
    print(sample_number)

    df_spatial = df_spatial_temp
    print(len(df_spatial_temp))
    # df_na = df_spatial.loc[np.isnan(df_spatial[sample_number])]
    for shift in shift_array:
        df_spatial_1 = df_spatial_temp.copy(deep=True)
        df_spatial_1['X'] = df_spatial_1['X'].add(shift[0])
        df_spatial_1['Y'] = df_spatial_1['Y'].add(shift[1])
        df_spatial_1.rename(index=str, columns={sample_number: "SHIFT_{0}".format(i)}, inplace=True)
        df_spatial_1.set_index(['X', 'Y'])
        df_spatial.set_index(['X', 'Y'])
        # print(df_spatial)
        # print(df_spatial_1)
        # df_spatial = df_spatial.join(df_spatial_1, how = 'left')#
        # print(df_spatial_1)
        df_spatial = df_spatial.merge(df_spatial_1, how='left', on=['X', 'Y'])
        i = i + 1
        print(i)
    df_spatial[['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']] = df_spatial[['SHIFT_0',
                                                                                                 'SHIFT_1',
                                                                                                 'SHIFT_2',
                                                                                                 'SHIFT_3',
                                                                                                 'SHIFT_4',
                                                                                                 'SHIFT_5']].fillna(
        value=empty_slot_value)
    df_spatial['COLUMN_MEAN_{0}'.format(sample_number)] = df_spatial[
        ['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']].mean(axis=1)
    df_spatial_adj = df_spatial[['X', 'Y', sample_number, 'COLUMN_MEAN_{0}'.format(sample_number)]]

    # df_stacked_temp = df_spatial_temp2[['X','Y',sample_number,'COLUMN_MEAN_{0}'.format(sample_number)]]
    df_spatial_adj.rename(index=str,
                          columns={sample_number: y_col, 'COLUMN_MEAN_{0}'.format(sample_number): "SURROUND_MEAN"},
                          inplace=True)
    # df_spatial_adj['SAMPLE_NUMBER'] = sample_number
    # print('lowess')
    # print(df_spatial_adj)

    df_spatial_adj = compute_lowess(df=df_spatial_adj,
                                    frac=frac,
                                    it=it,
                                    delta_factor=delta_factor,
                                    y_col=y_col,
                                    x_col='SURROUND_MEAN',
                                    y_pred_col=y_pred_col
                                    )

    # print(df_spatial_adj)
    chart_subname = sample_number
    if bool(number_to_name_dict):
        if str(sample_number) in number_to_name_dict:
            chart_subname = number_to_name_dict[str(sample_number)]
    if (save_plot):
        plot_lowess(df=df_spatial_adj,
                    y_col=y_col,
                    x_col='SURROUND_MEAN',
                    y_pred_col=y_pred_col,
                    OUTPUT_CHARTS_DIR=OUTPUT_CHARTS_DIR,
                    chart_subname=chart_subname)
    if (save_table):
        df_spatial_adj.to_csv(os.path.join(OUTPUT_CHARTS_DIR,'lowess_table_{0}.tsv'.format(chart_subname)), sep='\t', index=False)
    mean_value = np.nanmedian(df_spatial_adj[y_pred_col])

    df_spatial_adj[y_col] = round(df_spatial_adj[y_col] - df_spatial_adj[y_pred_col] + [mean_value] * len(df_spatial_adj.index), 3)
    df_spatial_adj.rename(index=str, columns={y_col: sample_number}, inplace=True)

    df_spatial_adj.drop([y_pred_col, 'SURROUND_MEAN'], axis=1, inplace=True)
    # df_spatial_adj = pd.concat([df_spatial_adj, df_na], ignore_index=True)
    df_spatial_adj.sort_values(by=['X', 'Y'], inplace=True)
    print(len(df_spatial_adj))
    # You need something to merge to so on the first loop it just creates a deep copy
    #  if(first_sample):
    #	   df_spatial_adj_all = df_spatial_adj.copy()
    #	  first_sample = False
    # df_stacked = pd.concat([df_stacked, df_stacked_temp], ignore_index = True)
    #	   continue
    #  df_spatial_adj_all = df_spatial_adj_all.merge(df_spatial_adj, how = 'outer', on =['X','Y'])

    # self.df_dict[df_in_name] = df_spatial_adj_all
    return df_spatial_adj[[sample_number]]


class spatial_correction:
    def local_spatial_correction_single(self, transformation_class):
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        descriptor_columns = transformation_class.descriptor_columns
        intensity_col = transformation_class.intensity_col
        y_pred_column = 'PREDICTED_INTENSITY'
        empty_slot_value = transformation_class.empty_slot_value
        save_plot = transformation_class.save_plot
        OUTPUT_CHARTS_DIR = transformation_class.OUTPUT_CHARTS_DIR
        import pandas as pd
        if len(list(set(['X', 'Y']).intersection(set(self.df_dict[df_in_name].columns)))) < 2:
            print('Missing X, Y columns needed for spatial analysis.')
            return
        if not descriptor_columns:
            descriptor_columns = list(set(self.descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))
        else:
            descriptor_columns = list(set(descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))

        import pandas as pd
        sample_array = self.data_columns
        sample_array.sort()
        first_sample = True
        df_spatial_adj_all = pd.DataFrame()

        for sample_number in sample_array:
            # if check for column names before crashing

            df_spatial = self.df_dict[df_in_name][['X', 'Y', sample_number]]
            shift_array = [[0, 2], [0, -2], [1, 1], [-1, 1], [1, -1], [-1, -1]]
            i = 0
            df_spatial_temp = df_spatial.copy(deep=True)
            for shift in shift_array:
                df_spatial_1 = df_spatial_temp.copy(deep=True)
                df_spatial_1['X'] = df_spatial_1['X'].add(shift[0])
                df_spatial_1['Y'] = df_spatial_1['Y'].add(shift[1])
                df_spatial_1.rename(index=str, columns={sample_number: "SHIFT_{0}".format(i)}, inplace=True)
                # print(df_spatial_1)
                df_spatial_1.set_index(['X', 'Y'])
                df_spatial.set_index(['X', 'Y'])
                df_spatial = df_spatial.join(df_spatial_1, how='left')  # , on= ['X','Y'])
                # df_spatial = df_spatial.merge(df_spatial_1, how = 'left', on= ['X','Y'])
                i = i + 1
                print(i)
            df_spatial[['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']] = df_spatial[['SHIFT_0',
                                                                                                         'SHIFT_1',
                                                                                                         'SHIFT_2',
                                                                                                         'SHIFT_3',
                                                                                                         'SHIFT_4',
                                                                                                         'SHIFT_5']].fillna(
                value=empty_slot_value)
            df_spatial['COLUMN_MEAN_{0}'.format(sample_number)] = df_spatial[
                ['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']].mean(axis=1)
            df_spatial_adj = df_spatial[['X', 'Y', sample_number, 'COLUMN_MEAN_{0}'.format(sample_number)]]
            # df_stacked_temp = df_spatial_temp2[['X','Y',sample_number,'COLUMN_MEAN_{0}'.format(sample_number)]]
            df_spatial_adj.rename(index=str, columns={sample_number: intensity_col,
                                                      'COLUMN_MEAN_{0}'.format(sample_number): "SURROUND_MEAN"},
                                  inplace=True)
            # df_spatial_adj['SAMPLE_NUMBER'] = sample_number
            # print('lowess')
            # print(df_spatial_adj)
            df_spatial_adj = compute_lowess(df=df_spatial_adj,
                                            frac=0.3,
                                            it=3,
                                            delta_factor=0.01,
                                            y_col=intensity_col,
                                            x_col='SURROUND_MEAN',
                                            y_pred_col=y_pred_column
                                            )

            # print(df_spatial_adj)
            if (save_plot):
                plot_lowess(df=df_spatial_adj,
                            y_col=intensity_col,
                            x_col='SURROUND_MEAN',
                            y_pred_col=y_pred_column,
                            OUTPUT_CHARTS_DIR=OUTPUT_CHARTS_DIR,
                            chart_subname=sample_number)
            df_spatial_adj[intensity_col] = round(df_spatial_adj[intensity_col] - df_spatial_adj[y_pred_column], 3)
            df_spatial_adj.rename(index=str, columns={intensity_col: sample_number}, inplace=True)
            df_spatial_adj.drop([y_pred_column, 'SURROUND_MEAN'], axis=1, inplace=True)
            # You need something to merge to so on the first loop it just creates a deep copy
            if (first_sample):
                df_spatial_adj_all = df_spatial_adj.copy()
                first_sample = False
                # df_stacked = pd.concat([df_stacked, df_stacked_temp], ignore_index = True)
                continue
            df_spatial_adj_all = df_spatial_adj_all.merge(df_spatial_adj, how='outer', on=['X', 'Y'])
        df_spatial_adj_all = df_spatial_adj_all.merge(self.df_dict[df_in_name][descriptor_columns], how='inner',
                                                      on=['X', 'Y'])
        self.df_dict[df_out_name] = df_spatial_adj_all
        return

    def compute_multi_lowess_self(df_spatial_temp, frac):  # , it, delta_factor, y_col, x_col, ypred_col):
        # import multiprocessing as mp
        # import os
        # from functools import partial
        # print(constant_dict)
        # frac =constant_dict['frac']
        # it =constant_dict['it']
        # delta_factor=constant_dict['delta_factor']
        # y_col =constant_dict['y_col']
        # x_col = constant_dict['x_col']
        # y_pred_col =constant_dict['y_pred_col']
        import pandas as pd
        shift_array = [[0, 2], [0, -2], [1, 1], [-1, 1], [1, -1], [-1, -1]]
        i = 0

        sample_number = list(set(df_spatial_temp.columns) - {'X', 'Y'})[0]
        print(sample_number)
        # df_spatial_temp = df_spatial.copy(deep=True)
        for shift in shift_array:
            df_spatial_1 = df_spatial_temp.copy(deep=True)
            df_spatial_1['X'] = df_spatial_1['X'].add(shift[0])
            df_spatial_1['Y'] = df_spatial_1['Y'].add(shift[1])
            df_spatial_1.rename(index=str, columns={sample_number: "SHIFT_{0}".format(i)}, inplace=True)
            # print(df_spatial_1)
            df_spatial = df_spatial.merge(df_spatial_1, how='left', on=['X', 'Y'])
            i = i + 1
            print(i)

        df_spatial[['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']] = df_spatial[['SHIFT_0',
                                                                                                     'SHIFT_1',
                                                                                                     'SHIFT_2',
                                                                                                     'SHIFT_3',
                                                                                                     'SHIFT_4',
                                                                                                     'SHIFT_5']].fillna(
            value=empty_slot_value)
        df_spatial['COLUMN_MEAN_{0}'.format(sample_number)] = df_spatial[
            ['SHIFT_0', 'SHIFT_1', 'SHIFT_2', 'SHIFT_3', 'SHIFT_4', 'SHIFT_5']].mean(axis=1)
        df_spatial_adj = df_spatial[['X', 'Y', sample_number, 'COLUMN_MEAN_{0}'.format(sample_number)]]
        # df_stacked_temp = df_spatial_temp2[['X','Y',sample_number,'COLUMN_MEAN_{0}'.format(sample_number)]]
        df_spatial_adj.rename(index=str, columns={sample_number: intensity_col,
                                                  'COLUMN_MEAN_{0}'.format(sample_number): "SURROUND_MEAN"},
                              inplace=True)
        # df_spatial_adj['SAMPLE_NUMBER'] = sample_number
        # print('lowess')
        # print(df_spatial_adj)
        df_spatial_adj = compute_lowess(df=df_spatial_adj,
                                        frac=0.3,
                                        it=3,
                                        delta_factor=0.01,
                                        y_col=intensity_col,
                                        x_col='SURROUND_MEAN',
                                        y_pred_col=y_pred_column
                                        )

        # print(df_spatial_adj)
        if (save_plot):
            plot_lowess(df=df_spatial_adj,
                        y_col=intensity_col,
                        x_col='SURROUND_MEAN',
                        y_pred_col=y_pred_column,
                        OUTPUT_CHARTS_DIR=OUTPUT_CHARTS_DIR,
                        chart_subname=sample_number)
            df_spatial_adj.to_csv('/scratch/{0}.txt'.format(sample_number), sep = '\t', index=False)
        df_spatial_adj[intensity_col] = round(df_spatial_adj[intensity_col] - df_spatial_adj[y_pred_column], 3)
        df_spatial_adj.rename(index=str, columns={intensity_col: sample_number}, inplace=True)
        df_spatial_adj.drop([y_pred_column, 'SURROUND_MEAN'], axis=1, inplace=True)
        # You need something to merge to so on the first loop it just creates a deep copy
        #  if(first_sample):
        #	   df_spatial_adj_all = df_spatial_adj.copy()
        #	  first_sample = False
        # df_stacked = pd.concat([df_stacked, df_stacked_temp], ignore_index = True)
        #	   continue
        #  df_spatial_adj_all = df_spatial_adj_all.merge(df_spatial_adj, how = 'outer', on =['X','Y'])
        df_spatial_adj_all = df_spatial_adj_all.merge(self.df_dict[df_in_name][descriptor_columns], how='inner',
                                                      on=['X', 'Y'])
        self.df_dict[df_in_name] = df_spatial_adj_all
        return

    def local_spatial_correction(self, transformation_class):
        import pandas as pd
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        descriptor_columns = transformation_class.descriptor_columns
        intensity_col = transformation_class.intensity_col
        y_pred_col = 'PREDICTED_INTENSITY'
        empty_slot_value = transformation_class.empty_slot_value
        save_plot = transformation_class.save_plot
        save_table = transformation_class.save_table
        OUTPUT_CHARTS_DIR = transformation_class.OUTPUT_CHARTS_DIR

        if len(list(set(['X', 'Y']).intersection(set(self.df_dict[df_in_name].columns)))) < 2:
            print('Missing X, Y columns needed for spatial analysis.')
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy()
            return
        if not descriptor_columns:
            descriptor_columns = list(set(self.descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))
        else:
            descriptor_columns = list(set(descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))

        import os
        from functools import partial
        import pandas as pd
        sample_array = self.data_columns
        sample_array.sort()
        first_sample = True
        df_spatial_adj_all = pd.DataFrame()
        cpu_count = os.cpu_count()
        max_count = len(sample_array)
        constant_dict2 = {'frac': 0.3,
                          'it': 3,
                          'delta_factor': 0.01,
                          'y_col': intensity_col,
                          'x_col': 'SURROUND_MEAN',
                          'y_pred_col': y_pred_col}
        # print(constant_dict2)
        compute_multi_lowess_map = partial(compute_multi_lowess, frac=0.3,
                                           it=3,
                                           delta_factor=0.01,
                                           y_col=intensity_col,
                                           x_col='SURROUND_MEAN',
                                           y_pred_col=y_pred_col,
                                           empty_slot_value=empty_slot_value,
                                           OUTPUT_CHARTS_DIR=OUTPUT_CHARTS_DIR,
                                           save_plot=save_plot,
                                           save_table = save_table,
                                           number_to_name_dict = self.number_to_name_dict)  # constant_dict = constant_dict2)

        # df_spatial_adj_array =
        df_spatial_adj = self.df_dict[df_in_name][['X', 'Y']]
        df_spatial_adj.sort_values(by=['X', 'Y'], inplace=True)
        #df_spatial_adj_array = [df_spatial_adj_array]
        for lower_sample_num in range(0, max_count, cpu_count):
            import multiprocessing as mp
            from contextlib import closing

            df_array = []
            upper_sample_num = lower_sample_num + cpu_count
            if lower_sample_num + cpu_count > max_count:
                upper_sample_num = max_count
            for sample_num_array in range(lower_sample_num, upper_sample_num, 1):
                df_spatial = self.df_dict[df_in_name][['X', 'Y', sample_array[sample_num_array]]]
                df_array.append(df_spatial)
            # print(df_array)
            with closing(mp.Pool(cpu_count)) as pool:
                # pool = mp.Pool(processes=cpu_count)
                df_array = pool.map(compute_multi_lowess_map, df_array)

            for df_i in df_array:
                colname_i = list( df_i.columns)[0]
            # print(df_spatial_adj_array)
                df_spatial_adj[colname_i] = list( df_i[colname_i] )

        df_spatial_adj = df_spatial_adj.merge(self.df_dict[df_in_name][descriptor_columns], how='inner',
                                                      on=['X', 'Y'])

        self.df_dict[df_out_name] = df_spatial_adj
        print('Finished Local Spatial Correction')

    # df_spatial_adj_all = df_spatial_adj_all.merge(df_spatial_adj, how = 'outer', on =['X','Y'])
    # df_spatial_adj_all = df_spatial_adj_all.merge(self.df_dict[df_in_name][descriptor_columns], how = 'inner', on =['X','Y'])
    # for sample_number in sample_array:
    # if check for column names before crashing
    # df_spatial = self.df_dict[df_in_name][['X','Y',sample_number]]

    def large_area_spatial_correction(self, transformation_class):
        import pandas as pd
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        descriptor_columns = transformation_class.descriptor_columns
        window_size = transformation_class.window_size
        import numpy as np
        from scipy import interpolate
        from scipy.interpolate import interp2d
        from scipy import ndimage
        if len(list(set(['X', 'Y']).intersection(set(self.df_dict[df_in_name].columns)))) < 2:
            print('Missing X, Y columns needed for spatial analysis.')
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy()
            return
        if not descriptor_columns:
            descriptor_columns = list(set(self.descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))
        else:
            descriptor_columns = list(set(descriptor_columns).intersection(set(self.df_dict[df_in_name].columns)))

        if df_in_name != df_out_name:
            self.df_dict[df_out_name] = self.df_dict[df_in_name][descriptor_columns]

        for data_column_i in self.data_columns:
            print(data_column_i)
            print(len(self.df_dict[df_in_name]))
            # Turn the XYZ to a 2D array like object that interp2d and uniform filter can handle
            array = self.df_dict[df_in_name].pivot_table(index='Y',
                                                         columns='X',
                                                         values=data_column_i,
                                                         aggfunc='mean').values
            # create a mask for the nan and replace them with a cubic interpolation

            x = np.arange(0, array.shape[1])
            y = np.arange(0, array.shape[0])
            array_np = np.ma.masked_invalid(array)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~array_np.mask]
            y1 = yy[~array_np.mask]
            newarr = array_np[~array_np.mask]

            GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                       (xx, yy),
                                       method='linear')
            GD1[np.isnan(GD1)] = np.nanmean(array)
            # apply a filter.  It seems like 75 works well there is no good rule of thumb (2% area?)
            GD2 = ndimage.uniform_filter(GD1, size=window_size)
            array_c = array - GD2
            # Go from a 2d matrix to X, Y, Intensity (Z) columns
            df_corrected = pd.DataFrame(array_c).stack().rename_axis(['Y', 'X']).reset_index(name=data_column_i)

            # There is an offset that needs to be corrected as the index goes back to zero
            df_corrected.X = df_corrected.X + self.df_dict[df_in_name]['X'].min()
            df_corrected.Y = df_corrected.Y + self.df_dict[df_in_name]['Y'].min()
            print('Min Correction: {0}'.format(np.nanmin(GD2)))
            print('Max Correction: {0}'.format(np.nanmax(GD2)))

            if df_in_name == df_out_name:
                # Drop the previous result
                self.df_dict[df_in_name].drop(columns=[data_column_i], inplace=True)
                # Use merge to replace the corrected result
                self.df_dict[df_in_name] = self.df_dict[df_in_name].merge(df_corrected, on=['X', 'Y'], how='left')
            else:
                # Use merge to add the corrected result
                self.df_dict[df_out_name] = self.df_dict[df_out_name].merge(df_corrected, on=['X', 'Y'], how='left')
            print(len(self.df_dict[df_out_name]))


class local_spatial_correction:
    def __init__(self,
                 df_in_name='df',
                 df_out_name=None,
                 descriptor_columns=[],
                 intensity_col='INTENSITY',
                 empty_slot_value=-3,
                 frac=0.3,
                 it=3,
                 delta=0.01,
                 save_plot=True,
                 save_table=False,
                 OUTPUT_CHARTS_DIR=None):
        self.transformation_name = 'local_spatial_correction'
        self.df_in_name = df_in_name
        self.df_out_name = df_out_name
        self.descriptor_columns = descriptor_columns
        self.intensity_col = intensity_col
        self.empty_slot_value = empty_slot_value
        self.frac = frac
        self.it = it
        self.delta = delta
        self.save_plot = save_plot
        self.save_table = save_table
        self.OUTPUT_CHARTS_DIR = OUTPUT_CHARTS_DIR
        return


class large_area_spatial_correction:
    def __init__(self,
                 df_in_name='df',
                 df_out_name='df',
                 descriptor_columns=[],
                 window_size=75,
                 save_plot=True,
                 OUTPUT_CHARTS_DIR=None):
        self.transformation_name = 'large_area_spatial_correction'
        self.df_in_name = df_in_name
        self.df_out_name = df_out_name
        self.descriptor_columns = descriptor_columns
        self.window_size = window_size
        self.save_plot = save_plot
        self.OUTPUT_CHARTS_DIR = OUTPUT_CHARTS_DIR
        return
