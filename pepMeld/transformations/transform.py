from .utils import *
from .basic import *
from .spatial_correction import *
from .clustering import *
# import .spatial_correction
# import .clustering
from .open_files import *
# from .open_files import open_file
class df_files(utils_transformations, basic_transformations, spatial_correction, clustering):
	import os
	import pandas
	def __init__(self,
				 file_type = None, # required
				 filepaths = [], # required
				 meta_filepaths = [], # required* (if you want to use it)
				 column_lookup={}, #optional --> needed only if columns differ from defaults
				 data_column_class = default_data_columns(),
				 required_meta_columns = default_meta_file().required_descriptor_columns,
				 logtransformed = False,
				 median_transformed = False,
				 meta_sep='\t',
				 data_sep='\t'): #optional to add more columns to metafile by first creating a object and saving over
		self.file_type = file_type
		self.filepaths = filepaths
		self.meta_filepaths = meta_filepaths
		self.meta_sep = meta_sep
		self.data_sep = data_sep
		# self.files_exist = list(map(os.path.isfile, self.filepaths))
		# self.files_meta_exist = list(map(os.path.isfile, self.meta_filepaths))
		self.required_meta_columns = required_meta_columns
		self.sample_name_column = required_meta_columns[1]
		self.vendor_name_column = required_meta_columns[0]
		self.rename_dictionary = {}
		self.data_column_class = data_column_class
		self.descriptor_columns = set()
		self.old_data_columns = []
		self.data_columns = []
		self.number_to_name_dict = {}
		# self.df = pd.DataFrame()
		# self.df_medians = pd.DataFrame()
		self.df_dict = {'df':pd.DataFrame(),
						'medians':pd.DataFrame()
					   }
		
		# skewed probes sequences by the sequence id.
		# These have a variation that is high by an arbitrarily set threshold.
		# Threshold was +/- 0.75 fold change, and created a peptide to be > or <  than 1.5 fold change
		self.df_skewed = pd.DataFrame()
		#done on orgininal data
		self.logtransformed = logtransformed
		#done on orgininal data
		self.median_transformed = median_transformed
		self = open_data(t_class=self)
