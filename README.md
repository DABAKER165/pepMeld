# pepMeld
Peptide Microarray Workflow. Python 3.5. docker, open-source.

# Peptide Microarray Overview
- Peptide microarrays are lawns of peptides tethered to a chip to measure the binding affinity of various immune system components
- Serum containing, antibodies or major histocompatibility complex exposed to the chip, and then washed to leave only binding immune components.
- These immune components are flourescently tagged, and the intensity flourecensce response is measured for each peptide lawn.
- Applications in eptope discovery, antibody responses, allergy profiling, and t-cell responses.

# pepMeld Summary.
- This workflow is focused on Nimble Therapeutics peptide microarray technology, though it can be concievably used and applied to other peptide microarray technology.
- It takes raw data as recieved from the vendor and processes the data through a series of transforms and filters.
- These steps are controlled through a json formatted file, and can be chosen depending upon what is important.

# pepMeld Workflow
- The workflow is designed to be reformated and flexible.  
- Much of the options could be left as default if using Nimble Therapeutics data (as of August 2020.  

## pepMeld Workflow Outline
- Logrithmic Transform
- Cluster Analysis for Defects
- Local Spatial Correction for signal bleed
- Large Area Spatial Correction
- Median per peptide probe for repeats
- Baseline Matching
- Background Subtraction / Sample Subtraction
- Rolling Medians
- Rolling Min/Max
- Stacking (a.k.a. Melting) data
- Merging Sequence files (a.k.a corresponding keys)
- Merging Sample Metadata files
- Merging Sequence Metadata files
- CSV output at any step to analyze intermediate files (custom separators can be declared, i.e. '\t', ',' )

## Tips and tricks in managing a lot of data
- The data output is set up to be readily visualized by an assorment of software as tsv files (i.e. matplotlib, plotly Excel etc) for fast statisitcial discovery and flipping through charts.
- For this reason, the file sizes can be several GB of diskspace and require over 16 GB of memory to process.
- Often times small variants of a peptides are added (like in different strains of a virust), Reducing the number of sequences per analysis, reducing the number of samples per analysis, or preprocessing the sequence files (known as corresponding keeys can help.
- The data is not effeciently minimized or compressed.  Tricks can be done to minimize data by 10-100 fold, but will need special software to extract, visualize, and analyze and has not been contained in the package.

## Configuration
Since every experiment and set up varies the configuration is primarily managed with a json file. Steps will need to be chosen base on the data recieved, and goals of the experiment.  Some steps may be important for some experiments, and not applicable for others.
- This configuration uses examples that have docker with mapped drives (-v) and a shortcut of /data_path.  You can replace optionally these file paths with your own.

## JSON Setup
- Each json file is an array of dictionary, (list of objects)
- Each dictionary is a "transformation", that transforms data, merges new data, or saves the data a intermediate file.
- There are 5 main objects of which are lowercase, and words are underscore delimited:
	- name (required, double quoted string)
		- must be unique and makes it easier to follow the output
	- order  (required, unquoted float / numeric)
		- must be unique and controls the order that the list of transfomrations are run
		- decimals up to 4 places and negatives are allows
	- skip (optional unquoted value of true  or  false )
		- default is false , must be lowercase and without quotes
		- set to true if you want to skip the step with out deleting it entirely (instead of comments
	- tranformation_func (required, double quoted string)
		- Case sensitive, must match a transformation on the list
	- transformation_args (required, dictionary object)
		- Must match the arguments of the transformation_func
		- Defaults vary depending upon the function

### Notes about dataframe in and outs
- every tranformation argument has a dataframe in and/or out name
- Df Inputs names must match a previous outputted dataframe
- df outputs can save over/ replace by using the same name (to save ram)
- df out puts can have a unique name as needed.

### Open Data and Meta data Files ( transformation_func : "open_data_meta_files" )
- transformation_func : "open_data_meta_files"
- opens a list or single file path
- Used to open the data files and the meta data about each sample, including possible background and column subtraction
- will concatenate opened files.
- Checks for required columns, keeps columns, and renames columns based on configuration
- accepts gzipped files  (.gz extention)
####  Open Files ( "open_data_meta_files" ) transformation_args:

- data_filepaths (required, double quoted list of strings OR string default =  None and will error) 
	- List or single filepath to open
- data_sep (required single character string, default = "\t")
	- set to the delimiter of the file i.e. "\t", "," etc
- meta_filepaths (required, double quoted list of strings OR string default =  None and will error) 
	- List or single filepath to open
- meta_sep (required single character string, default = "\t")
	- set to the delimiter of the file i.e. "\t", "," etc

- data_required_descriptor_columns (required, double quoted list of strings default = [])
	- Looks for these columns to make sure they exist in the data file
	- ORIGINAL name not the renamed.
- meta_required_descriptor_columns (required, double quoted list of strings default = [])
	- Looks for these columns to make sure they exist in the meta data file
	- ORIGINAL name not the renamed.
- data_default_descriptor_headers (required, double quoted list of strings default = [])
	- Descriptive data like X,Y position on the chip that is retained
- data_keep_columns (required, double quoted list of strings default = [])
	- Keeps only these columns (to save memory)
	- Default will keep all columns
- data_rename_dictionary (optional, dictionary, default = {})
	- if set, will rename columns accordingly, "original name" : "new name"
- sample_name_column (required, string, default= "SAMPLE_NAME")
	- Column of the sample name in the metadata file
	- Each row MUST have a unique name in the metadata file
- meta_vendor_name_column (required, string, default= "VENDOR_NAME")
	- The column names in the original data file
	- Could be same as SAMPLE_NAME
	- MUST be unique in the data file and metadata file
- meta_sample_number_column (required, string, default= "SAMPLE_NUMBER")
	- Used to make sure each row is unique value and minimize data through out steps
	- Muse be included in the metadata file


#### Open Files ( "open_data_meta_files" ) example:
```
{
		"name" : "open_data_meta_files",
		"order" : 0.1,
		"transformation_func":"open_data_meta_files",
		"transformation_args" : {
			"data_sep":"\t",
			"data_filepaths": "/data_path/Raw_aggregate_data.txt",
			"meta_filepaths":"/data_path/meta_all.csv",
			"meta_sep":","
			}
	}

```

### Open Files ( transformation_func : "open_files" )
- transformation_func : "open_files"
- opens a list or single file path
- will concatenate opened files.
- Checks for required columns, keeps columns, and renames columns based on configuration
- accepts gzipped files (.gz extention)
####  Open Files ( "open_files" ) transformation_args:

- filepaths (required, double quoted list of strings OR string default =  None and will error) 
	- List or single filepath to open
- required_descriptor_columns (required, double quoted list of strings default = [])
	- Looks for these columns to make sure they exist in the file
	- ORIGINAL name not the renamed.
- keep_columns (required, double quoted list of strings default = [])
	- Keeps only these columns (to save memory)
	- Default will keep all columns
- rename_dictionary (optional, dictionary, default = {})
	- if set, will rename columns accordingly, "original name" : "new name"
- df_out_name (required, double quoted string, default = None and will error)
- sep (required single character string, default = "\t")
	- set to the delimiter of the file i.e. "\t", "," etc

#### Open Files ( "open_files" ) example:
```
{
	"name" : "open_files corresponding file",
	"order" : 10.0,
	"transformation_func":"open_files",
	"transformation_args" : {
		"filepaths" : "/seq_path/sequence_file.tsv",
		"df_out_name" : "df_sequence_file",
		"required_descriptor_columns" : ["SEQ_ID","PROBE_SEQUENCE", "POSITION"],
		"keep_columns":["SEQ_ID", "SEQ_NAME", "PROTEIN", "POSITION", "PROBE_SEQUENCE"],
		"rename_dictionary":{"PROBE_SEQUENCE":"PROBE_SEQUENCE", "SEQ_ID":"SEQ_ID", "POSITION":"POSITION", "PROTEIN":"PROTEIN", "SEQ_NAME":"SEQ_NAME"},
		"sep":"\t"
	}
}

```

### Log Transform ( transformation_func : "log_transform" )
- transformation_func : "log_transform"
- Used take the log transform of all the data values.  Base two is recommended.
- Taking the log of the data will match the data closer to a normal distribution and is therefore recommended.

#### Log Transform ( "log_transform" ) transformation_args:

- base (optional, unquoted float / numeric, default = 2) 
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)

#### Log Transform ( "log_transform" ) example:
```
{
	"name" : "log_transform",
	"order" : 1,
	"transformation_func":"log_transform",
	"transformation_args" : {
		"base" : 2,
		"df_in_name" : "df",
		"df_out_name" : "df"
	}
}

```

### Find Clusters of Defects ( transformation_func : "find_clusters" )
- transformation_func : "find_clusters"
- Used to find clusters of defects on the array (physically/spatially)
- Use in conjuction withexclude_clustered_data, to exclude this data
- Uses scipy dbscan for the clusterin algorithm
- Automatically set to multithread.
- Must have X - Y coordinate columns labeled X and Y

#### Find Clusters of Defects ( "find_clusters" ) transformation_args:

- OUTPUT_CHARTS_DIR (required if save_plot = true, string)
	- driectory of where to ouput the cluster charts with save_plot = true
	- directory path must exist 
- save_plot (optional, true/false, default = true)
	- set to save plots of the defect clusters
- percentile_slices (optional, numeric/float, default 20)
	- how many slices to subset the data by (percentile) before looking for cluseters
	- 2 would split the data from 0-50, 25-75 and 50-100 percentile and look for clusters
	- 20 would split data. 0-5, 2.5-7.5, 5-10 ... 92.5 - 97.5 , 95-100 percentiles
- eps (optional, numeric/float, default 3)
	- distance to nearest neighbor same as scipy kit
- min_samples (optional, numeric/float, default 3)
	- How many peptide lawns within the cluster to be considered a cluster 
- merge_to_input_df (optional, true/false, default true)
	- leave as true to exclude later.
	- Set as false if you do not want to exclude data later with exclude_clustered_data
- df_in_name (required, double quoted string, default = None and will error)
- df_cluster_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)
#### Find Clusters of Defects ( "find_clusters" ) example:
```
{
	"name" : "Find Clusters and Merge with DF",
	"order" : 2.0,
	"transformation_func":"find_clusters",
	"transformation_args" : {
		"df_in_name":"df",
		"df_cluster_name":"expanded_cluster",
		"df_out_name":"df_clustered",
		"OUTPUT_CHARTS_DIR":"/data_path/out",
		"merge_to_input_df":true,
		"percentile_slices":20,
		"eps":3,
		"min_samples":10,
		"save_plot":true
	}
}

```

### Exclude Clusters of Defects ( "exclude_clustered_data" )
- transformation_func : "exclude_clustered_data"
- Used to exclude data from  find_clusters transform
- find_clusters must be ran first and have the associated df_in_name to df_cluster_name

#### Exclude Clusters of Defects ( "exclude_clustered_data" ) transformation_args:


- min_cluster_ratio (optional,numeric float, default = 0):
	- set to 0 is a check for if the cluster sizes is too small or large relative to the expanded region
- min_original_cluster_size (optional,numeric float, default = 10)
	- should match the min_samples of find_clusters but can be set higher if desired
- filter_expanded (optional, true/false, default true)
	- set to true if you want to filter clusters that are within an outer polygon of cluster
- filter_clusters (optional, nrue/false, default true)
	- set to true if you want to only fiter cluster data
- unstack (optional, true/false, default true)
	- set if the data is stacked or not
- df_in_name (required, double quoted string, default = None and will error)
	- should match the df_cluster_name of find_clusters
- df_out_name (required, double quoted string, default = None and will error)

#### Exclude Clusters of Defects ( "exclude_clustered_data" ) example:
```
{
		"name" : "Exclude clusters",
		"order" : 2.2,
		"transformation_func":"exclude_clustered_data",
		"transformation_args" : {
			"df_in_name":"df_clustered",			
			"df_out_name":"df",
			"unstack":true,
			"filter_clusters":true,
			"filter_expanded":true,
			"min_original_cluster_size":10,
			"min_cluster_ratio":0
		}
	}

```
### Local Spatial correction ( transformation_func : "local_spatial_correction" )
- transformation_func : "local_spatial_correction"
- Used to compensate for signal bleed
- Plots for each peptide lawn, plots the surrounding lawns mean value by the target lawn value
- Then calculates a lowess prediction
- Then subtracts the difference from the median of the lowess prediction
- Reduces the effect of a large intensity adjacent to a lawn from inflating its value
- Plots the lowess predictions to show the significances of the correction

#### Local Spatial correction( "local_spatial_correction" ) transformation_args:

- empty slot value (required, numeric/floag, default = 2)
	- Intensity value to apply to a adjacent lawn with no peptides
- OUTPUT_CHARTS_DIR (required if save_plot = true, string)
	- driectory of where to ouput the cluster charts with save_plot = true
	- directory path must exist 
- save_plot (optional, true/false, default = true)
	- set to save plots of the defect clusters
- save_table (optional, true/false, default = true)
	- set to the data table of the lowess predicted values with the actual values
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)

#### Local Spatial correction ( "local_spatial_correction" ) example:
```
{
		"name" : "local_spatial_correction",
		"order" : 3.0,
		"transformation_func":"local_spatial_correction",
		"transformation_args" : {
			"df_in_name":"df",			
			"df_out_name":"df",
			"empty_slot_value":2,
			"save_plot":true,
			"save_table":true,
			"OUTPUT_CHARTS_DIR":"/data_path/out"
		}
	}	

```

### Large Area Spatial Correction ( transformation_func : "large_area_spatial_correction" )
- transformation_func : "large_area_spatial_correction"
- Corrects for intensity gradiant across a large physical area of the slide
- Must have X - Y coordinate columns labeled X and Y

#### Large Area Spatial Correction ( "large_area_spatial_correction" ) transformation_args:
- OUTPUT_CHARTS_DIR (required if save_plot = true, string)
	- driectory of where to ouput the cluster charts with save_plot = true
	- directory path must exist 
- save_plot (optional, true/false, default = true)
	- set to save plots of the defect clusters
- window_size : (required, numeric/float, default =75)
	- side of the square (i.e. 75 x 75) the the gradiant is measured on
	- Larger gradiant for subltle changes, smaller for larger changes
	- Setting very small, (i.e. under 10) will simply disrupt the data without reasoning
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)

#### Large Area Spatial Correction ( "large_area_spatial_correction" ) example:
```
{
	"name" : "large_area_spatial_correction",
	"order" : 4.0,
	"skip" : false,
	"transformation_func":"large_area_spatial_correction",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name":"df",
		"window_size":75,
		"save_plot":true,
		"OUTPUT_CHARTS_DIR":"/data_path/out"
	}
}

```



### Subract Column ( transformation_func : "subtract_columns" )
- transformation_func : "subtract_columns"
- Configured in a merged metadata file from open_data_meta_files transformation_func
- Add a column to the metadata table, and put the SAMPLE_NAME of what to be subtracted for each sample row
- Only works on unstacked data.

#### Subract Column ( "subtract_columns" ) transformation_args:

- subtract_column (required, quoted string, default = None and will error)
	- Column with SAMPLE_NAMES that whose value will be used to subtract
- sample_name : (required, quoted string, default = None and will error)
	- Column to be refrenced for the SAMPLE_NAME and starting value
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)

#### Subract Column( "subtract_columns" ) example:
```
{
	"name" : "Subtract 0 DPI",
	"order" : 6.5,
	"transformation_func":"subtract_columns_class",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name":"df_subtract",
		"subtract_column" : "DPI_SUBTRACT",
		"sample_name" : "SAMPLE_NAME"
	}
}	

```

### Median grouped by list of columns ( transformation_func : "median_group_by" )
- transformation_func : "median_group_by"
- Takes the median of all the data columns grouped by a list of columns

#### Median grouped by list of columns ( "median_group_by" )transformation_args:
- perform on unstacked /unmelted data
- group_by_columns (required, quoted array (comma delimited brackets), default = None) 
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)


#### example:

```
{
	"name" : "Take Median of Replicates",
	"order" : 5.0,
	"transformation_func":"median_group_by",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name":"df",
		"group_by_columns" : ["PROBE_SEQUENCE","PEP_LEN","EXCLUDE"]
	}
}
```

### Rolling median of df ( transformation_func : "rolling_median" )
- transformation_func : "rolling_median"
- Used to take a rolling median by a window (typically 3)
- Helps "smoothen data" from flyers or erroneous points
- Often done grouped by SAMPLE_NAME and SEQ ID (protein level)
- Must choose a sort order often ["POSITION","SAMPLE_NAME","SEQ_ID"]
- Works on stacked data by default or data_stacked= false for unstacked data
#### Rolling median of df( "rolling_median" ) transformation_args:

- group_by_columns (required, double quoted list of strings, default = None and will error) 
	- What to Groups to apply the rolling mean to.
- sort_by_columns (required, double quoted list of strings, default = None and will error)
	- Sorts by the list in the order left to right
- rolling_window (optional, unquoted float/numeric, default = 3)
- value_column (string)
	- Set to the column you want to do the rolling_median on
	- if "data_stacked" : false then it will perform on all data columns and ignore this flag
- data_stacked (optional, unquoted true / false, default = true)
	- set with a value column if true, value
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)

#### Rolling median of df ( "rolling_median" ) example:
```
{
	"name" : "Rolling median of df",
	"order" : 10.15,
	"transformation_func":"rolling_median",
	"transformation_args" : {
		"df_in_name":"df_stacked",			
		"df_out_name" : "df_stacked",
		"group_by_columns" : ["SEQ_ID","SAMPLE_NAME"],
		"sort_by_columns" : ["POSITION","SAMPLE_NAME","SEQ_ID"],
		"rolling_window" : 3,
		"value_column" : "INTENSITY",
		"data_stacked" : true
	}
}
{
	"name" : "Rolling median of df",
	"order" : 10.15,
	"transformation_func":"rolling_median",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name" : "df",
		"group_by_columns" : ["SEQ_ID"],
		"sort_by_columns" : ["POSITION","SEQ_ID"],
		"rolling_window" : 3,
		"data_stacked" : false
	}
}
``` 


### Shift baseline to percentile ( transformation_func : "shift_baseline" )
- transformation_func : "shift_baseline"
- Shifts the baseline to a set percentile to match the baseline of the data
- Helps compare data without disrupting the min/maxes.
- Percentile should be chosen with in the baseline of the analyzed samples

#### Shift baseline to percentile ( "shift_baseline" )transformation_args:
- percentile (required, unquoted float/numeric, default = 25) 
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)


#### example:

```
{
	"name" : "Shift to the 25 percentile",
	"order" : 6.0,
	"transformation_func":"shift_baseline",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name":"df",
		"percentile" : 25
	}
}
```

### Merge /Join data ( transformation_func : "merge_data_frames" )

- transformation_func : "merge_data_frames"
- enacts the pandas "merge" function to the data
- by default will merge on all same name columns.  Override by declaring on

#### Stack / Melt data ( transformation_func : "melt_class" ) transformation_args:

- how ( optional , default = "inner")
- on (optional, double quoted list of strings, default = all same name columns)
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)


#### example:
```
{
	"name" : "Merge Stack Data with Meta Data",
	"order" : 8.0,
	"transformation_func":"merge_data_frames",
	"transformation_args" : {
		"df_name_left":"meta",			
		"df_name_right":"stacked",
		"df_out" : "stacked",
		"how" : "inner"
	}
}
```


### Stack / Melt data ( transformation_func : "melt_class" )
- transformation_func : "melt_class"
- enacts the pandas "melt" function to the data
- value_name of "INTENSITY" is recommended
- all data columns will be selected by default unless otherwise specified
- var_name, value_vars and id_vars will autoselect to proper columns if not included.

#### Stack / Melt data ( transformation_func : "melt_class" ) transformation_args:
- value_name (required, unquoted float/numeric, default = 25) 
- var_name ( optional , default = null and will auto select)
- df_in_name (required, double quoted string, default = None and will error)
- df_out_name (required, double quoted string, default = None and will error)
- value_vars (optional, autoselects if set to default = []),
- id_vars = (optional, autoselects if set to default = [])

#### example:
```
{
	"name" : "Stack Data",
	"order" : 7.0,
	"transformation_func":"melt_class",
	"transformation_args" : {
		"df_in_name":"df",			
		"df_out_name":"stacked",
		"value_name" : "INTENSITY",
		"var_name":null
	}
}
```


# Getting Started
- I reccommend the docker container.  It will make sure that Python and the required libraries are correctly loaded to compatibile versions.  You can also load a a standalone, but will be at the mercy of the incompatible versions due to outdated or updated software.


## Running with docker.
- You will need to map drives to access data outside of the docker.  
1. download the the pepMeld_docker.tar.gz file
2. untar the file
3. change to the directory of the untarred folder
4. build the docker image from the docker file
5. run the docker image with mapped drives
6. run pepMeld with the location of the json tranformation file
```


# change to the pepMeld_docker folder that was downloaded and untarred
cd ~/pepMeld/pepMeld_docker
# Build with docker
docker build -t pepmeld:v1 .

# Run with docker with paths declared as desired
docker run --cpus 4 -it -v ~/pepMeld/sequences/:/seq_path -v ~/pepMeld/data/:/data_path pepmeld:v1

# Inside the docker container (note the interactive mode here) run the pyscript
# This can be done out side of interactive mode as well
python process_arrays.py --config=/data_path/transformations_v1.json

```

# Addiing on your own transforms
- If you do not see a transform you like, you can add your own. The program is fairly modular, but will need an import command for your files you add.
- Compiling with docker:
```
# Change to the directory of pepMeld you made changes to
cd ~/pepMeld/

# Tar the folder
tar -czvf pepMeld.tar.gz pepMeld

# Move the folder to  the docker folder
mv -f ~/pepMeld/pepMeld.tar.gz ~/pepMeld/pepMeld_docker/

# change to the pepMeld_docker folder that was downloaded and untarred
cd ~/pepMeld/pepMeld_docker
# Build with docker
docker build -t pepmeld:v1 .

# Run with docker with paths declared as desired
docker run --cpus 4 -it -v ~/pepMeld/sequences/:/seq_path -v ~/pepMeld/data/:/data_path pepmeld:v1

# Inside the docker container (note the interactive mode here) run the pyscript
# This can be done out side of interactive mode as well
python process_arrays.py --config=/data_path/transformations_v1.json

```


# References
AK, et al. (2020) “High-Throughput Identification of MHC Class I Binding Peptides Using an Ultradense Peptide Array.” J Immunol. 204(6): 1689-96. doi: 10.4049/jimmunol.1900889

Heffron AS, et al. (2018) "Antibody responses to Zika virus proteins in pregnant and non-pregnant macaques." PLoS NTD. 12(11): e0006903. doi: 10.1371/journal.pntd.0006903

Bailey A, et al. (2017) "Pegivirus avoids immune recognition but does not attenuate acute-phase disease in a macaque model of HIV infection." PLoS Pathogens. 13(10): e1006692. doi: 10.1371/journal.ppat.1006692

https://www.nimbletherapeutics.com/technology
