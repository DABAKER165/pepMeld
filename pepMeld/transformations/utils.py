import pandas as pd

"""These are utilities that open files, writes files"""


class utils_transformations:
	def merge_data_frames(self, transformation_class):
		df_out = transformation_class.df_out
		df_name_left = transformation_class.df_name_left
		df_name_right = transformation_class.df_name_right
		on_columns = transformation_class.on_columns
		how = transformation_class.how
		if not on_columns or len(on_columns) < 1:
			on_columns = list(set(self.df_dict[df_name_left].columns).intersection(set(self.df_dict[df_name_right].columns)))
		self.df_dict[df_out] = self.df_dict[df_name_left].merge(self.df_dict[df_name_right], on=on_columns, how= how)
		print("merged data frames: " + df_name_left + " & " + df_name_right + " into " + df_out)
		
	def melt_df(self, transformation_class):
		df_in_name = transformation_class.df_in_name
		df_out_name = transformation_class.df_out_name
		value_vars = transformation_class.value_vars
		id_vars = transformation_class.id_vars
		value_name = transformation_class.value_name
		var_name = transformation_class.var_name
		if var_name is None:
			var_name = self.sample_name_column
		
		if not id_vars:
			id_vars = list(set(self.df_dict[df_in_name].columns) - set(self.data_columns ))	
		if not value_vars:
			value_vars = self.data_columns
		self.df_dict[df_out_name] = pd.melt(self.df_dict[df_in_name], 
											 id_vars = id_vars, 
											 value_vars = value_vars, 
											 var_name = var_name, 
											 value_name = value_name)
		print('Stacked data using: \n id_vars = [' + ','.join(id_vars) + ' ]')
		print('Stacked data using: \n value_vars = [' + ','.join(value_vars) + ' ]')
		print('Stacked data using:   var_name = ' + var_name + ', value_name= ' + value_name)
		
	def add_dataframe(self, df, df_name):
		if df_name in self.df_dict.keys():
			print('Information: Saving Existing Dataframe.')
		self.df_dict[df_name] = df
		print('Manually added DataFrame to data_class: ' + df_name)
		
	def remove_dataframe(self, df_name):
		if df_name in self.df_dict.keys():
			del self.df_dict[df_name]
			print('Removed Dataframe: ' + df_name)
		else:
			print('WARNING: Did not Remove Dataframe because it does not exist: ' + df_name)

	def save_to_csv(self,transformation_class):

		df_in_name = transformation_class.df_in_name 
		filepath = transformation_class.filepath
		sep = transformation_class.sep
		index  = transformation_class.index

		if len(self.number_to_name_dict) > 0:
			self.df_dict[df_in_name].rename(columns=self.number_to_name_dict, inplace=True)
			self.df_dict[df_in_name].to_csv(filepath, sep=sep, index=index)
			rename_back = {v: k for k, v in self.number_to_name_dict.items()}
			self.df_dict[df_in_name].rename(columns=rename_back, inplace=True)
		else:
			self.df_dict[df_in_name].to_csv(filepath, sep=sep, index=index)
		print('Outputted to data file | filepath: ' + df_in_name + ' | ' + filepath)
		
class save_to_csv:
	def __init__(self,
				 df_in_name = None,
				 filepath = None,
				 sep = '\t',
				 index  = False
				):
		self.transformation_name = 'save_to_csv'
		self.df_in_name = df_in_name
		self.filepath = filepath
		self.sep = sep
		self.index = index


class melt_class:
	def __init__(self,
				 df_in_name = None,
				 df_out_name = None,
				 value_name = 'INTENSITY',
				 var_name= None,
				 value_vars = [],
				 id_vars = []
				):
		self.transformation_name = 'melt_df'
		self.df_in_name = df_in_name
		self.df_out_name = df_out_name
		self.value_vars = value_vars
		self.id_vars = id_vars
		self.value_name = value_name
		self.var_name = var_name
		
class merge_data_frames:
	def __init__(self,
				 df_name_left= None, 
				 df_name_right= None, 
				 df_out = None,
				 on_columns=[], 
				 how = 'inner'):
		self.transformation_name = 'merge_data_frames'
		self.df_name_left = df_name_left
		self.df_name_right= df_name_right
		self.df_out = df_out
		self.on_columns = on_columns
		self.how = how	   
