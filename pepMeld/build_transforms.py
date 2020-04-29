# build_tranforms

from .add_transformation_classes import transformation_class
from .transformations.basic import *
# from .transformations.clustering import *
# from .transformations.spatial_correction import *
from .transformations.transform import *
from .transformations.utils import *


def build_transforms(args):
	transformations = transformation_class()
	################################################
	#	 Pre treatment of the Data.	#
	################################################

	transformations.add_transformation(name = 'log_transform',
									  order = 1,
									  transformation = log_transform(base = args['log_base'],
													   df_in_name = 'df',
													   df_out_name = 'df'))

	transformations.add_transformation(name = 'log_transform to tsv',
									  order = 1.1,
									  skip = False,
									  transformation = save_to_csv(df_in_name = 'df',
																		 filepath = os.path.join(args['data_output_dir'],'df_log_transform.tsv'),
																		 sep = args['output_sep'],
																		 index  = False))

	transformations.add_transformation(name='Find Clusters and Merge with DF',
									   order=2.0,
									   skip=False,
									   transformation=find_clusters(df_input_name='df',
																	df_cluster_name='expanded_cluster',
																	df_output_name='df_clustered',
																	OUTPUT_CHARTS_DIR=args['charts_output_dir'],
																	merge_to_input_df=True,
																	percentile_slices=20,
																	eps=3,
																	min_samples=10,
																	save_plot=True))
	transformations.add_transformation(name='save df_clustered to tsv',
									   order=2.1,
									   skip=False,
									   transformation=save_to_csv(df_in_name='df_clustered',
																  filepath=os.path.join(args['data_output_dir'],'df_clustered.tsv'),
																  sep='\t',
																  index=False))
	# Filter criteria
	# Un Stack
	transformations.add_transformation(name='exclude clusters',
									   order=2.2,
									   skip=False,
									   transformation=exclude_clustered_data(df_in_name='df_clustered',
																			 df_out_name='df',
																			 descriptor_columns=[],
																			 unstack=True,
																			 filter_clusters=True,
																			 filter_expanded=True,
																			 min_original_cluster_size=10,
																			 min_cluster_ratio=0))

	transformations.add_transformation(name='save df_excluded_clusters to tsv',
									   order=2.3,
									   skip=False,
									   transformation=save_to_csv(df_in_name='df',
																  filepath=os.path.join(args['data_output_dir'],'df_excluded_clustered.tsv'),
																  sep='\t',
																  index=False))



	################################################
	#	 CONTROL_SUBTRACT  #
	################################################
	transformations.add_transformation(name = 'control_subtract',
									  order = 3,
									   skip = False,
									  transformation = subtract_columns_class(subtract_column = 'CONTROL_SUBTRACT',
																				sample_name= 'SAMPLE_NAME',
																				subtract_dict = None,
																				df_out_name = 'df',
																				df_in_name = 'df'))

	################################################
	#	 Take Median of data   #
	################################################

	transformations.add_transformation(name = 'median',
									   skip = False,
									  order = 5,
									  transformation = median_group_by(group_by_columns = ['PROBE_SEQUENCE','PEP_LEN','EXCLUDE'],
																				 df_out_name = 'df',
																	   df_in_name = 'df'))


	################################################
	#	Stack Median Data  #
	################################################
	transformations.add_transformation(name = 'stack_data',
									  order = 6.1,
									  transformation = melt_class(value_name = 'INTENSITY',
																	  var_name= None,
																	  value_vars = [],
																	  id_vars = [],
																	  df_out_name = 'stacked',
																	  df_in_name = 'df'))
	################################################
	#	MERGE META and CORR KEY Data  #
	################################################

	transformations.add_transformation(name = 'merge_stacked',
									  order = 7.0,
									  skip = False,
									  transformation = merge_data_frames(df_name_left = 'meta',
																			 df_name_right = 'stacked',
																			 df_out = 'stacked',
																			 on_columns=[],
																			 how = 'inner'))
	################################################
	#  Merge Corr key and 0 DPI Subtract #
	################################################
	transformations.add_transformation(name = 'merged_stacked_corr_meta',
									  order = 8.1,
									  skip = False,
									  transformation = merge_data_frames(df_name_left = 'corr_merged',
																			 df_name_right = 'stacked',
																			 df_out = 'stacked',
																			 on_columns=[],
																			 how = 'inner'))

	transformations.add_transformation(name='median to tsv',
									   order=8.2,
									   skip=False,
									   transformation=save_to_csv(df_in_name='df',
																  filepath=os.path.join(args['data_output_dir'],
																						'median_log.tsv'),
																  sep=args['output_sep'],
																  index=False))
	transformations.add_transformation(name='meta to tsv',
									   order=6.3,
									   skip=False,
									   transformation=save_to_csv(df_in_name='meta',
																  filepath=os.path.join(args['data_output_dir'],
																						'meta.tsv'),
																  sep=args['output_sep'],
																  index=False))
	transformations.add_transformation(name='stacked to tsv',
									   order=6.4,
									   skip=False,
									   transformation=save_to_csv(df_in_name='stacked',
																  filepath=os.path.join(args['data_output_dir'],
																						'stacked.tsv'),
																  sep=args['output_sep'],
																  index=False))

	transformations.add_transformation(name = 'merged_stacked_corr to tsv',
						  order = 8.3,
						  skip = False,
						  transformation = save_to_csv(df_in_name = 'stacked',
															 filepath = os.path.join(args['data_output_dir'] , 'merged_stacked_meta_seq.tsv'),
															 sep = args['output_sep'],
															 index  = False))
	transformations.add_transformation(name = 'merged_stacked to tsv',
						  order = 7.1,
						  skip = False,
						  transformation = save_to_csv(df_in_name = 'stacked',
															 filepath = os.path.join(args['data_output_dir'], 'merged_stacked.tsv'),
															 sep = args['output_sep'],
															 index  = False))
	return transformations
