import os 
import pandas as pd

class default_data_columns:
	import os
	def __init__(self, 
				 default_descriptor_headers = {'PEP_LEN', 'REPL', 'SEQ_ID', 'PROBE_SEQUENCE', 'PROBE_ID', 'POSITION','X','Y'},
				 keep_columns = {'PEP_LEN', 'REPL', 'PROBE_SEQUENCE', 'X', 'Y'},
				 required_descriptor_columns = {'PROBE_SEQUENCE'}
					
				):
		self.default_descriptor_headers = default_descriptor_headers
		self.keep_columns = keep_columns
		self.required_descriptor_columns = required_descriptor_columns


class default_meta_file:
	import os
	def __init__(self, 
					required_descriptor_columns = ['VENDOR_NAME','SAMPLE_NUMBER','SAMPLE_NAME']  
				):
		self.required_descriptor_columns = required_descriptor_columns
		

class default_corr_file:
	import os
	def __init__(self, 
				 default_descriptor_headers = {'PEPTIDE_SEQUENCE', 'ORIGINAL_POSITION', 'VIRUS_ACCENSION', 'PROTIEN_ACCESSION', 'VIRUS', 'PROTIEN','ORIGINAL_SEQ_ID'},
				 keep_columns = {'PEPTIDE_SEQUENCE', 'ORIGINAL_POSITION', 'VIRUS_ACCENSION', 'PROTIEN_ACCESSION', 'VIRUS', 'PROTIEN','ORIGINAL_SEQ_ID','VIRUS_TYPE'},
				 required_descriptor_columns = {'PEPTIDE_SEQUENCE','ORIGINAL_SEQ_ID','ORIGINAL_POSITION'}, 
				 rename_dictionary = {'PEPTIDE_SEQUENCE':'PROBE_SEQUENCE', 'ORIGINAL_SEQ_ID':'SEQ_ID', 'ORIGINAL_POSITION':'POSITION'}
				):
		self.default_descriptor_headers = default_descriptor_headers
		self.keep_columns = keep_columns
		self.required_descriptor_columns = required_descriptor_columns
		self.rename_dictionary = rename_dictionary
		if not set(rename_dictionary.keys()).issubset(required_descriptor_columns):
			raise RuntimeError("Corr rename dictionary missing columns: " + "\n".join(list(rename_dictionary.keys())) + ' out of required columns : ' + "\n".join(list(required_descriptor_columns))+ '\n Extend Required columns or shorten rename list \n Script has terminated')


class default_corr_all_file:
	import os
	def __init__(self, 
				 default_descriptor_headers = {'SEQ_ID','PROBE_SEQUENCE', 'POSITION'},
				 keep_columns = {'SEQ_ID','PROBE_SEQUENCE', 'POSITION'},
				 required_descriptor_columns = {'SEQ_ID','PROBE_SEQUENCE', 'POSITION'}, 
				 rename_dictionary = {'PROBE_SEQUENCE':'PROBE_SEQUENCE', 'SEQ_ID':'SEQ_ID', 'POSITION':'POSITION'}
				):
		self.default_descriptor_headers = default_descriptor_headers
		self.keep_columns = keep_columns
		self.required_descriptor_columns = required_descriptor_columns
		self.rename_dictionary = rename_dictionary
		if not set(rename_dictionary.keys()).issubset(required_descriptor_columns):
			raise RuntimeError("Corr rename dictionary missing columns: " + "\n".join(list(rename_dictionary.keys())) + ' out of required columns : ' + "\n".join(list(required_descriptor_columns))+ '\n Extend Required columns or shorten rename list \n Script has terminated')

class default_protein_lookup_file:
	import os
	def __init__(self, 
				 default_descriptor_headers = {'SEQ_ID','START_POSITION','END_POSITION','PROTEIN'},
				 keep_columns = {'SEQ_ID','START_POSITION','END_POSITION','PROTEIN'},
				 required_descriptor_columns = {'SEQ_ID','START_POSITION','END_POSITION','PROTEIN'},
				 rename_dictionary = {}
				):
		self.default_descriptor_headers = default_descriptor_headers
		self.keep_columns = keep_columns
		self.required_descriptor_columns = required_descriptor_columns
		self.rename_dictionary = rename_dictionary
		if not set(rename_dictionary.keys()).issubset(required_descriptor_columns):
			raise RuntimeError("Corr rename dictionary missing columns: " + "\n".join(list(rename_dictionary.keys())) + ' out of required columns : ' + "\n".join(list(required_descriptor_columns))+ '\n Extend Required columns or shorten rename list \n Script has terminated')
			
class default_virus_lookup_file:
	import os
	def __init__(self, 
				 default_descriptor_headers = {'SEQ_ID','VIRUS','VIRUS_TYPE'},
				 keep_columns = {'SEQ_ID','VIRUS','VIRUS_TYPE','VIRUS_ACCESSION'},
				 required_descriptor_columns = {'SEQ_ID','VIRUS'},
				 rename_dictionary = {}
				):
		self.default_descriptor_headers = default_descriptor_headers
		self.keep_columns = keep_columns
		self.required_descriptor_columns = required_descriptor_columns
		self.rename_dictionary = rename_dictionary
		if not set(rename_dictionary.keys()).issubset(required_descriptor_columns):
			raise RuntimeError("Corr rename dictionary missing columns: " + "\n".join(list(rename_dictionary.keys())) + ' out of required columns : ' + "\n".join(list(required_descriptor_columns))+ '\n Extend Required columns or shorten rename list \n Script has terminated')


def check_filepaths_exists(filepath_list):
	if not filepath_list: # check to see if they are not empty
		print('warning: no filepaths must be declared')
		return
	missing_filepaths = [x for x in filepath_list if not os.path.isfile(x)]
	if len(missing_filepaths) > 0:	
		raise RuntimeError('Files do not exist: ' + '\n'.join(missing_filepaths) + ' ; Script has terminated')


def check_columns(filepath, sep, required_columns):
	df_colnames = pd.read_csv(filepath, nrows=0, sep = sep).columns.tolist()
	missing_required_columns  = list(set(required_columns) - set(df_colnames))
	if len(missing_required_columns) > 0 :   
		print("Missing Required Columns in: " + filepath + "\n".join(missing_required_columns))
		raise RuntimeError("Missing Required Columns in: " + filepath + "\n".join(missing_required_columns) + ' ; Script has terminated')
	return df_colnames


def get_DataFrame_Gen(self):
	check_filepaths_exists(filepath_list=self.filepaths)
	df = pd.DataFrame()
	for filepath_i in self.filepaths:
		df_colnames = check_columns(filepath=filepath_i, sep=self.key_all_sep, required_columns=self.default_class.required_descriptor_columns)
		usecols = list(self.default_class.keep_columns.intersection(set(df_colnames)))
		df_i = pd.read_csv(filepath_i, sep = self.key_all_sep, usecols= usecols)
		df = pd.concat([df,df_i], ignore_index = True)	
	self.old_columns = df.columns
	df.rename(columns=self.default_class.rename_dictionary, inplace=True)
	self.new_columns = df.columns
	self.df = df


def open_data(t_class):
	df = pd.DataFrame()
	df_meta = pd.DataFrame()
	print(t_class.filepaths)
	check_filepaths_exists(filepath_list=t_class.filepaths)
	for filepath_i in t_class.meta_filepaths:
		df_colnames = check_columns(filepath=filepath_i, sep=t_class.meta_sep, required_columns=t_class.required_meta_columns)
		# usecols = list(t_class.default_class.keep_columns.intersection(set(df_colnames)))
		df_meta_i = pd.read_csv(filepath_i, sep = t_class.meta_sep)
		df_meta = pd.concat([df_meta,df_meta_i], ignore_index = True)
		# exclude rows as needed
		if 'EXCLUDE' in df_meta.columns:
			df_meta['EXCLUDE'] = df_meta.EXCLUDE.astype(str)
			df_meta = df_meta.loc[df_meta.EXCLUDE != 'EXCLUDE']
		# make sure the type of column is str
		df_meta[t_class.sample_name_column] = df_meta[t_class.sample_name_column].astype(str)
		t_class.rename_dictionary = dict(zip(df_meta[t_class.vendor_name_column], df_meta[t_class.sample_name_column]))
		if ("SAMPLE_NAME" in list(df_meta.columns)) and  ("SAMPLE_NUMBER" in list(df_meta.columns)):
			t_class.number_to_name_dict = dict(zip( df_meta['SAMPLE_NUMBER'], df_meta["SAMPLE_NAME"]))
		# save the meta file
		t_class.df_dict['meta'] = df_meta

	data_columns = set()
	for filepath_i in t_class.filepaths:
		df_colnames_i = check_columns(filepath=filepath_i, sep=t_class.data_sep, required_columns=t_class.data_column_class.required_descriptor_columns)
		# print(df_colnames_i)
		data_columns_i = set(df_colnames_i) - set(t_class.data_column_class.default_descriptor_headers)
		# print(data_columns_i)
		data_columns_i = data_columns_i.intersection(t_class.rename_dictionary.keys())
		# print(data_columns_i)
		keep_columns = t_class.data_column_class.keep_columns.intersection(df_colnames_i)
		# print(keep_columns)
		on_columns = list(t_class.descriptor_columns.intersection(keep_columns))
		# print(on_columns)
		t_class.descriptor_columns =  t_class.descriptor_columns.union(keep_columns)
		# make sure two sets of data columns do not have the same named column
		duplicate_column_names = list(data_columns_i.intersection(data_columns))
		if duplicate_column_names:
			print("Duplicate named columns: " + filepath_i + "\n".join(duplicate_column_names))
		usecols_list = list(keep_columns.union(data_columns_i))
		df_i = pd.read_csv(filepath_i, sep=t_class.data_sep, usecols= usecols_list)
		data_columns = data_columns.union(data_columns_i)
		# join to existing or create a new one if it doesn't exist

		if len(df.index)>0:
			df = df.merge(df_i, how = 'outer', on= on_columns)
		else:
			df = df_i
		# rename the file with correct naming convenstion
		t_class.old_data_columns = list(data_columns)
		df.rename(columns=t_class.rename_dictionary, inplace=True)
		t_class.data_columns  =list(map(t_class.rename_dictionary.get, t_class.old_data_columns))
		# t_class.data_columns = lisdf.columns.intersection(data_columns_new)
		t_class.df_dict['df'] = df
		print(t_class.old_data_columns)
		print(t_class.data_columns)
		print(df)
	return t_class


class corr_files:
	import os
	import pandas as pd
	def __init__(self,
				 filepaths = [],
				 default_class = default_corr_file(),
				 key_all_sep = ','
				):
		self.filepaths = filepaths
		self.files_exist = list(map(os.path.isfile, self.filepaths))
		self.default_class = default_class
		# self.rename_dictionary=rename_dictionary
		self.df = pd.DataFrame()
		self.old_columns = []
		self.new_columns = []
		self.key_all_sep = key_all_sep
	# def get_DataFrames(self):
		get_DataFrame_Gen(self)
		
		
		
class corr_all_files:
	import os
	import pandas
	def __init__(self,
				 filepaths = [],
				 default_class = default_corr_all_file(),
				 key_all_sep = ','
				):
		self.filepaths = filepaths
		self.files_exist = list(map(os.path.isfile, self.filepaths))
		self.default_class = default_class
		# self.rename_dictionary=rename_dictionary
		self.df = pd.DataFrame()
		self.old_columns = []
		self.new_columns = []
		self.key_all_sep = key_all_sep
	# def get_DataFrames(self):
		get_DataFrame_Gen(self)
		self.df = self.df.loc[self.df['SEQ_ID'] != 'REDUNDANT']
		
class protein_lookup_files:
	def __init__(self,
				 filepaths = [],
				 default_class = default_protein_lookup_file(),
				 key_all_sep = '\t',
				 compressed = True
				):
		self.filepaths = filepaths
		self.files_exist = list(map(os.path.isfile, self.filepaths))
		self.default_class = default_class
		self.df= pd.DataFrame()
		self.old_columns = []
		self.new_columns = []
		self.key_all_sep = key_all_sep
		self.compressed = compressed 
		get_DataFrame_Gen(self)
		if self.compressed:
			df_protein = pd.DataFrame()
			for index, row in self.df.iterrows():
				df_i=pd.DataFrame({'SEQ_ID':row['SEQ_ID'],'PROTEIN':row['PROTEIN'] , 'POSITION':list(range(row['START_POSITION'],row['END_POSITION'] )) })
				df_protein = pd.concat([df_protein,df_i], ignore_index=True)
			self.df = df_protein
			
class virus_lookup_files:
	def __init__(self,
				 filepaths = [],
				 default_class = default_virus_lookup_file(),
				 key_all_sep = '\t',
				 compressed = True
				):
		self.filepaths = filepaths
		self.files_exist = list(map(os.path.isfile, self.filepaths))
		self.default_class = default_class
		self.df= pd.DataFrame()
		self.old_columns = []
		self.new_columns = []
		self.key_all_sep = key_all_sep
		self.compressed = compressed
		get_DataFrame_Gen(self)
