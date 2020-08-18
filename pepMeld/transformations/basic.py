class basic_transformations:
    def shift_baseline(self, transformation_class=None):
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        percentile = transformation_class.percentile

        if transformation_class is None:
            print('WARNING: NO TRANSFORMATION PASSED')
            return
        transform_columns = transformation_class.transform_columns
        if len(transformation_class.transform_columns) < 1:
            transform_columns = self.data_columns

        if df_out_name != df_in_name:
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy(deep=True)
        if percentile < 0 or percentile > 100:
            print('percentile is not between 0 and 1')
            return
        else:
            q = percentile/100
            self.df_dict[df_out_name][transform_columns] = self.df_dict[df_in_name][transform_columns] - self.df_dict[df_in_name][transform_columns].quantile(q=q)
        print('Data percentile shifted ' + str(transformation_class.percentile) + 'of each data column. Transformed on: [' + ', '.join(
            transform_columns) + ' ]')
        return

    def log_transformation(self, transformation_class=None):
        df_in_name = transformation_class.df_in_name
        df_out_name = transformation_class.df_out_name
        base = transformation_class.base
        if transformation_class is None:
            print('WARNING: NO TRANSFORMATION PASSED')
            return
        transform_columns = transformation_class.transform_columns
        if not transformation_class.transform_columns:
            transform_columns = self.data_columns


        import numpy as np
        if df_out_name != df_in_name:
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy(deep=True)
        if base == 2:
            self.df_dict[df_out_name][transform_columns] = self.df_dict[df_in_name][transform_columns].apply(
                np.log2)
        elif base == 10:
            self.df_dict[df_out_name][transform_columns] = self.df_dict[df_in_name][transform_columns].apply(np.log)
        else:
            self.df_dict[df_out_name][transform_columns] = self.df_dict[df_in_name][transform_columns].apply(
                np.log2)
            self.df_dict[df_out_name][transform_columns] = self.df_dict[df_in_name][transform_columns].div(
                np.log2(base))

        print('Data log ' + str(transformation_class.base) + ' Transformed on: [' + ', '.join(
            transform_columns) + ' ]')


    def median_group_by(self, transformation_class):
        group_by_columns = transformation_class.group_by_columns
        df_out_name = transformation_class.df_out_name
        df_in_name = transformation_class.df_in_name
        missing_group_by_columns = list(set(group_by_columns) - self.descriptor_columns)
        if missing_group_by_columns:
            print('WARNING: Group by columns missing: ' + '\n'.join(missing_group_by_columns))
        group_by_columns = list(self.descriptor_columns.intersection(set(group_by_columns)))
        if not group_by_columns:
            print('WARNING: groupby columns all missing median transform did not take place: ' + ';'.join(
                group_by_columns))
            return
        if df_out_name == df_in_name:
            print('Warning: Future median evaluations cannot take place-output and input same.')
        self.df_dict[df_out_name] = self.df_dict[df_in_name].groupby(group_by_columns, as_index=False)[
            self.data_columns].median()
        print('Median grouped by: [' + ', '.join(group_by_columns) + ' ]')

    def get_subtract_columns_dictionary(self,
                                        subtract_column='CONTROL_SUBTRACT',
                                        sample_name='SAMPLE_NAME'):
        # self.sample_name_column
        if (subtract_column not in self.df_dict['meta'].columns) or (sample_name not in self.df_dict['meta'].columns):
            error_message = ''
            if subtract_column not in self.df_dict['meta'].columns:
                error_message = ' subtract_column missing: ' + subtract_column + ' ; '
            if sample_name not in self.df_dict['meta'].columns:
                error_message = error_message + ' sample_name missing: ' + sample_name + ' ; '
            raise RuntimeError(
                "Missing Required Column in Meta Data: " + error_message + ' \n Out of Meta data columns: ' + ", ".join(
                    self.df_dict['meta'].columns) + ' ; \n Script has terminated')
        subtract_name_dict = {key: list(value)[0] for key, value in
                              self.df_dict['meta'].groupby(sample_name)[self.sample_name_column]}
        subtract_dict = {
        ';;'.join([subtract_name_dict[y] for y in key.split(';')]): [subtract_name_dict[x] for x in value] for
        key, value in self.df_dict['meta'].groupby(subtract_column)[sample_name]}
        return subtract_dict

    def subtract_columns(self, transformation_class):
        def correct_multiple_controls_3(CONTROL_1, CONTROL_2, max_diff=1):
            import numpy as np
            if np.isnan([CONTROL_1, CONTROL_2]).any():
                return np.nanmax([CONTROL_1, CONTROL_2])
            if abs(CONTROL_1 - CONTROL_2) > max_diff:
                return min([CONTROL_1, CONTROL_2]) + max_diff / 2
            else:
                return (CONTROL_1 + CONTROL_2) / 2

        # if self.background_subtracted and subtract_name is None:
        # print('Warning: Background already subtracted')
        # return
        subtract_dict = transformation_class.subtract_dict
        df_out_name = transformation_class.df_out_name
        df_in_name = transformation_class.df_in_name
        if transformation_class.subtract_dict is None:
            subtract_dict = self.get_subtract_columns_dictionary(subtract_column=transformation_class.subtract_column,
                                                                 sample_name=transformation_class.sample_name)

        if not subtract_dict:
            print('WARNING: empty background_subtract_dictionary, backround not subtracted')
        # if not background_subtract_dictionary:
        # background_subtract_dictionary = self.background_subtract_dictionary
        # backkground_column is the key
        # column_list is the list that is subtracted
        # Copy the dataframe if it is not saving over it
        if df_out_name != df_in_name:
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy(deep=True)
        for subtract_column, column_list in subtract_dict.items():
            column_list = list(set(self.data_columns).intersection(set(column_list)))
            if not column_list:
                print('Nothing to subtract: No columns in column list in data columns:' + ';'.join(column_list))
                continue
            if ';;' in subtract_column:
                print('DOUBLE_COLUMN')
                subtract_column_list = subtract_column.split(';;')
                print(subtract_column_list)
                # print(self.df_dict)
                self.df_dict[df_in_name][subtract_column] = self.df_dict[df_in_name].apply(
                    lambda row: correct_multiple_controls_3(row[subtract_column_list[0]],
                                                            row[subtract_column_list[1]]), axis=1)

            if subtract_column in self.df_dict[df_in_name].columns:
                self.df_dict[df_out_name][column_list] = self.df_dict[df_in_name][column_list].sub(
                    self.df_dict[df_in_name][subtract_column], axis=0)
            # self.data_columns.extend(column_list_append)
            else:
                print('Nothing Done Background columns donot exist: ' + subtract_column)
                continue
        # self.background_subtracted = True
        print('Background subtracted using:')
        print(subtract_dict)

    def rolling_median(self, transformation_class):
        rolling_window = transformation_class.rolling_window
        df_out_name = transformation_class.df_out_name
        df_in_name = transformation_class.df_in_name
        group_by_columns = transformation_class.group_by_columns
        sort_by_columns = transformation_class.sort_by_columns
        data_stacked = transformation_class.data_stacked
        data_columns = self.data_columns
        if data_stacked:
            data_columns = ['INTENSITY']
        if df_out_name != df_in_name:
            self.df_dict[df_out_name] = self.df_dict[df_in_name].copy(deep=True)
        print('rolling median on data columns')
        print(data_columns)
        self.df_dict[df_in_name].sort_values(by=sort_by_columns, ascending=True, inplace=True)
        self.df_dict[df_out_name][data_columns] = self.df_dict[df_in_name].groupby(group_by_columns)[
            data_columns].transform(lambda x: x.rolling(rolling_window).median())

        print('Applied rolling median of window ' + str(rolling_window) + ' to ' + ', '.join(self.data_columns))

    def evaluate_medians(self):
        # still working on a method here
        if len(self.df_dict['medians'].index) == 0:
            print('WARNING: No median data on file')
            return
        if len(self.df_dict['medians'].index) == len(self.df_dict['df'].index):
            print('WARNING: Median Data same as Data')
            return
        df = self.df

        df_melt = pd.melt(df, id_vars='PROBE_SEQUENCE', value_vars=self.data_columns)
        group_by_columns = ['PROBE_SEQUENCE', 'variable']
        df_melt['rank_value'] = df_melt.groupby(group_by_columns, as_index=False)['value'].rank()
        df_2_melt = df_melt.loc[df_melt['rank_value'] == 2]
        df_2_melt.rename(columns={'value': 'value_2'}, inplace=True)
        df_2_melt.drop(['rank_value'], axis=1, inplace=True)
        df_3_melt = df_melt.loc[df_melt['rank_value'] == 3]
        df_3_melt.rename(columns={'value': 'value_median'}, inplace=True)
        df_3_melt.drop(['rank_value'], axis=1, inplace=True)
        df_4_melt = df_melt.loc[df_melt['rank_value'] == 4]
        df_4_melt.rename(columns={'value': 'value_4'}, inplace=True)
        df_4_melt.drop(['rank_value'], axis=1, inplace=True)
        df_melt = df_2_melt.merge(df_3_melt, how='inner', on=group_by_columns)
        df_melt = df_melt.merge(df_4_melt, how='inner', on=group_by_columns)
        df_melt['value_4_median'] = df_melt['value_4'].sub(df_melt['value_median'], axis=0)
        df_melt['value_2_median'] = df_melt['value_2'].sub(df_melt['value_median'], axis=0)
        df_melt_skewed = df_melt.loc[
            ((df_melt['value_4'] > 1.5) & (df_melt['value_median'] < 1.5) & (df_melt['value_4_median'] > 0.75)) |
            ((df_melt['value_2'] < 1.5) & (df_melt['value_median'] > 1.5) & (df_melt['value_2_median'] < -0.75))]
        # df_melt_skewed['value_4_adjusted'] = value_median
        df_melt_skewed.sort_values(by='value_median', ascending=False, inplace=True)
        self.df_skewed = df_melt_skewed

# df_melt_skewed.to_csv(OUTPUT_TABLES_DIR + '/shiv_MHC_stacked_median_data_reduced.csv', index=False)
# print('outputed stacked to: ' + OUTPUT_TABLES_DIR + '/shiv_MHC_stacked_median_data_reduced.csv')


class median_group_by:
    def __init__(self,
                 group_by_columns=['PROBE_SEQUENCE', 'PEP_LEN'],
                 df_out_name='median',
                 df_in_name='df'):
        self.transformation_name = "median_group_by"
        self.group_by_columns = group_by_columns
        self.df_out_name = df_out_name
        self.df_in_name = df_in_name


class subtract_columns_class:
    def __init__(self,
                 subtract_column=None,
                 sample_name=None,
                 subtract_dict=None,
                 df_out_name=None,
                 df_in_name=None):
        self.transformation_name = "subtract_columns"
        self.subtract_column = None
        self.sample_name = None
        self.subtract_dict = None
        if subtract_dict is None:
            self.subtract_column = subtract_column
            self.sample_name = sample_name
        else:
            self.subtract_dict = subtract_dict
        self.df_out_name = df_out_name
        self.df_in_name = df_in_name


class rolling_median:
    def __init__(self,
                 rolling_window=3,
                 group_by_columns=None,
                 sort_by_columns=None,
                 df_out_name=None,
                 df_in_name=None,
                 data_stacked=True):
        self.transformation_name = "rolling_median"
        self.group_by_columns = group_by_columns
        self.sort_by_columns = sort_by_columns
        self.rolling_window = rolling_window
        self.df_out_name = df_out_name
        self.df_in_name = df_in_name
        self.data_stacked = data_stacked


class rolling_max_min:
    def __init__(self,
                 rolling_window=2,
                 df_out_name=None,
                 df_in_name=None):
        self.transformation_name = "rolling_max_min"
        self.rolling_window = rolling_window
        self.df_out_name = df_out_name
        self.df_in_name = df_in_name


class shift_baseline:
    def __init__(self,
                 percentile=25,
                 transform_columns=[],
                 df_in_name='df',
                 df_out_name='df'):
        self.transformation_name = 'shift_baseline'
        self.percentile = percentile
        self.transform_columns = transform_columns
        self.df_in_name = df_in_name
        self.df_out_name = df_out_name


class log_transform:
    def __init__(self,
                 base=2,
                 transform_columns=[],
                 df_in_name='df',
                 df_out_name='df'):
        self.transformation_name = 'log_transformation'
        self.base = base
        self.transform_columns = transform_columns
        self.df_in_name = df_in_name
        self.df_out_name = df_out_name
