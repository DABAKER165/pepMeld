
def get_max_font_width_height(string_list,font_name= 'DejaVuSans', font_size = 15):
    # = list(map(txt.measure,x_label_list))
    from PIL import ImageFont
    #unicode_text = 'DlD'
    font = ImageFont.truetype("/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/{0}.ttf".format(font_name), font_size)

    #print(string_list)
    max_string_width = max([font.getsize(x)[0] for x in string_list])
    max_string_height = max([font.getsize(x)[1] for x in string_list])
    return(max_string_width, max_string_height)

def set_plot_sizes(figure_size_pixels, sample_count, max_string_width, max_string_height, font_size):
    square_to_font = 0.5
    string_aspect_ratio = max_string_width/max_string_height
    square_height = figure_size_pixels[1] / (sample_count  + square_to_font * string_aspect_ratio)
    font_size_scaler = square_height * square_to_font / max_string_height
    #print(font_size_scaler)
    font_size_new = font_size * font_size_scaler * .5
    color_bar_proportion = int(figure_size_pixels[1] / (square_height + 1))
    scale_size = square_height ** 2 
    return(scale_size, font_size_new, color_bar_proportion)

def corr_heatmap(x, y, size,
            column_type,
            size_scale = 400, 
            figure_size_pixels = [6000,6000], 
            font_size = 'auto', 
            figure_dpi = 100,
            font_type= 'DejaVuSans'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    figure_size = [x/figure_dpi for x in figure_size_pixels]
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    
    
    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]
    
       
    sample_count = len(list(x.unique()))
    x_labels_TEMP = list(x.unique())
    y_labels_TEMP= list(y.unique())
    if column_type == 'str':
        x_labels_TEMP.sort()
        y_labels_TEMP.sort()
    else:
        x_labels_TEMP.sort(key=float)
        y_labels_TEMP.sort(key=float)
    # Mapping from column names to integer coordinates
    x_labels = [v for v in x_labels_TEMP]
    print(x_labels)
    y_labels = [v for v in y_labels_TEMP]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    print(x_to_num)
    #size_scale = size_scale
    
    if font_size == 'auto':
        max_string_width, max_string_height = get_max_font_width_height(string_list=x_labels, font_size = 32)
        size_scale, font_size_new, color_bar_proportion = set_plot_sizes(figure_size_pixels, sample_count, max_string_width, max_string_height, font_size= 32)
    else: 
        font_size_new = font_size
        color_bar_proportion = sample_count + 1
    print(font_size_new)
    SMALL_SIZE = font_size_new
    MEDIUM_SIZE = font_size_new + 2
    BIGGER_SIZE = font_size_new + 4

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE) 
    #plt.rcParams['font.sans-serif'] = font_type
    #plt.rc('ytick', labelsize=30) 
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams['savefig.dpi'] = figure_dpi
    fig, ax = plt.subplots()
  
    plot_grid = plt.GridSpec(1, color_bar_proportion, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:,:-1])
    print(color_bar_proportion)
    print('Here')
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_width, axis_height = bbox.width, bbox.height
    print(axis_width, axis_height)
    size_scale = (figure_dpi*axis_width * .9 * .8 /sample_count)**2
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size ** 3 * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s', # Use square as scatterplot marker
        color = size.apply(value_to_color),
        edgecolors='none'
    )
    x_max = len(x_to_num)-.5
    y_max = len(y_to_num)-.5
    ax.set_aspect(1)
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set(xlim=(-.5,x_max), ylim=(-.5,y_max))
    
    ax = plt.subplot(plot_grid[:,-1])
    col_x = [0]*len(palette) # Fixed x coordinate for the bars
    bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5]*len(palette), # Make bars 5 units wide
        left=col_x, # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False) # Hide grid
    ax.set_facecolor('white') # Make background white
    ax.set_xticks([]) # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()
    plt.show()
    
import matplotlib.colors as mcolors
def make_colormap(seq):
    
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb

class heat_map_x_axis_scale:
    def __init__(self,
                 first_last_label = 'first_and_last', #'last''first', 
                 spacing_multiple = 50
                ):
        self.first_last_label = first_last_label
        self.spacing_multiple = spacing_multiple
        return


class plot_heatmap_settings:
    def __init__(self,
                df = None,
                title = None,
                matrix_col_names = None,
                matrix_row_names = None,
                matrix_values = None,
                matrix_row_order = None,
                 aggfunc = 'mean',
                color_scale_dict = {},
                output_filepath = None,
                 group_row_by = None,
                 group_col_by = None,
                 col_range_dict = None,
                 row_range_dict = None,
                 custom_scheme = False,
                 color_scheme = 'magma',
                 sliver_size = 1,
                 color_range = [0,6],
                 font_size = 15,
                 spacing_multiple = None,
                 first_last_label = 'first_and_last',
                 extension = ['.png','.eps']
                ):
        self.df = df
        self.title = title
        self.matrix_col_names = matrix_col_names
        self.matrix_row_names = matrix_row_names
        self.matrix_values = matrix_values
        self.matrix_row_order = matrix_row_order
        self.color_scale_dict = color_scale_dict
        self.output_filepath = output_filepath
        self.group_row_by = group_row_by
        self.group_col_by = group_col_by
        self.col_range_dict = col_range_dict
        self.color_scheme = color_scheme
        self.custom_scheme = custom_scheme
        self.sliver_size = sliver_size
        self.color_range = color_range
        self.font_size = font_size
        self.spacing_multiple = spacing_multiple
        self.first_last_label = first_last_label
        self.aggfunc = aggfunc
        self.extension = extension
        
def plot_heatmap(settings):
    import matplotlib
    import matplotlib.cm as cm
    import numpy as np
    import numpy.random
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import seaborn as sns
    keep_column_list = [settings.matrix_col_names, 
                        settings.matrix_row_names, 
                        settings.matrix_values, 
                        settings.group_row_by,
                        settings.group_col_by]
    #remove nones 
    if settings.matrix_row_order:
        df_order = settings.df[[settings.matrix_row_order,settings.matrix_row_names]].drop_duplicates()
        df_order = df_order.groupby(settings.matrix_row_names)[settings.matrix_row_order].min().reset_index()
        #print(df_order)
    keep_column_list = [x for x in keep_column_list if x is not None]

    df = settings.df[keep_column_list].drop_duplicates()

    #sliver_size
    #if not settings.col_range_dict:
    col_range_dict = {}
    row_range_dict = {}
    col_labels = False
    if settings.group_col_by:   
        col_range_dict = {key: sorted(list(set(value[settings.matrix_col_names]))) for key,value in df.groupby(settings.group_col_by)}
        count_cols_df = df[[settings.group_col_by,settings.matrix_col_names]].drop_duplicates()
        count_cols_df['MIN'] = count_cols_df.groupby(settings.group_col_by)[settings.matrix_col_names].transform(min)
        count_cols_df = count_cols_df.groupby([settings.group_col_by,'MIN'])[settings.matrix_col_names].count().reset_index()
        count_cols_df['RATIO_WIDTH'] = count_cols_df[settings.matrix_col_names] /  count_cols_df[settings.matrix_col_names].max() 
        count_cols_df.sort_values('MIN',inplace=True)

        col_labels = True
    else:
        col_range_dict['A'] = sorted(list(set(df[settings.matrix_col_names])))
        #print(sorted(list(set(settings.df[settings.matrix_col_names]))))
        count_cols_df = pd.DataFrame({'RATIO_WIDTH':[1], 'A':'A'})
        settings.group_col_by = 'A'
        
    
    #row_range_dict = {key: sorted(list(set(value[settings.matrix_row_names]))) for key,value in df.groupby(settings.group_row_by)}
    #row_range_dict ={}
    row_labels = False
    if settings.group_row_by: 
        row_range_dict = {key: sorted(list(set(value[settings.matrix_row_names]))) for key,value in df.groupby(settings.group_row_by)}
        keep_column_list_temp = [settings.group_row_by,settings.matrix_row_names]#, settings.matrix_row_order]
        keep_column_list_temp = [x for x in keep_column_list if x is not None]
        count_rows_df = df[keep_column_list_temp].drop_duplicates()

        count_rows_df = count_rows_df.groupby([settings.group_row_by])[settings.matrix_row_names].count().reset_index() 
        count_rows_df['RATIO_WIDTH'] = count_rows_df[settings.matrix_row_names] /  count_rows_df[settings.matrix_row_names].max() 

        row_labels = True
        #print(count_rows_df)

    else:
        row_range_dict['A'] = sorted(list(set(settings.df[settings.matrix_row_names])))
        count_rows_df = pd.DataFrame({'RATIO_WIDTH':[1], 'A':'A'})
        settings.group_row_by = 'A'
        
        
    #df[settings.matrix_col_names] = df[settings.matrix_col_names] - ([settings.matrix_col_names] % settings.sliver_size )
    #df[settings.matrix_col_names] = 
    

    #sns.set(font_scale=2) 
    plot_rows = 1
    if settings.group_row_by:
        plot_rows =  len(count_rows_df.index)
    plot_cols = 1
    if settings.group_col_by:
        plot_cols = len(count_cols_df.index)
    #plot_rows = 1  

    #cmap = plt.cm.copper
    #norm = matplotlib.colors.Normalize(vmin= data.min(), vmax= data.max())
    
    #get a dictionary of the column separations
    width_ratios_list = count_cols_df['RATIO_WIDTH'].tolist()
    height_ratios_list = count_rows_df['RATIO_WIDTH'].tolist()
    #width_ratios_list.append(0.08)
    gridspec_kw = {"height_ratios":height_ratios_list, "width_ratios" : width_ratios_list,'wspace' :.02, 'hspace' :.2}
    plt.rc('font', size=settings.font_size) 
    #plt.rcParams.update({'font.size': settings.font_size})
    
    if not settings.custom_scheme:
        cmap = plt.cm[settings.color_scheme]
    else:
        cmap = settings.color_scheme
    heatmapkws = dict(square=False,   vmin= settings.color_range[0], vmax= settings.color_range[1], cmap = cmap)
    tickskw =  dict(xticklabels=False, yticklabels=False)
    norm = matplotlib.colors.Normalize(vmin= settings.color_range[0], vmax= settings.color_range[1])
    left = 0.13; right=0.87
    bottom = 0.2; top = 0.9

    fig, axis = plt.subplots(plot_rows,plot_cols,gridspec_kw=gridspec_kw)
    print(plot_rows)
    print(plot_cols)
    fig.subplots_adjust(left=left, right=right,bottom=bottom, top=top )

    axis_num = 0
    n_row = 0
    # loop through the rows
    for row_index, row_rows in count_rows_df.iterrows():
        n_col = 0
        #loop through the columns
        for index, col_rows in count_cols_df.iterrows(): 
            #filter the data for the matrix cell 
            df_temp = df.loc[df[settings.matrix_col_names].isin(col_range_dict[col_rows[settings.group_col_by]]) & df[settings.matrix_row_names].isin(row_range_dict[row_rows[settings.group_row_by]])]
            #Need to make sure we get the remainder of the sliver size as to have a partial cell filled and start at the given cell
            max_value = df_temp[settings.matrix_col_names].max() 
            min_value = df_temp[settings.matrix_col_names].min()
            if settings.sliver_size is not None:
                min_value_remainder = df_temp[settings.matrix_col_names].min() % settings.sliver_size
                
                # Assign the x possition to be the same for each sliver
                df_temp[settings.matrix_col_names] = df_temp[settings.matrix_col_names] - (df_temp[settings.matrix_col_names] % settings.sliver_size ) + min_value_remainder
                #print( df_temp)
                #find the median for each sliver

                df_temp = df_temp.groupby([settings.matrix_col_names,settings.matrix_row_names])[settings.matrix_values].mean().reset_index()

            #print( df_temp)
            #filter for the rows
            df_ordertemp = df_order[df_order[settings.matrix_row_names].isin(row_range_dict[row_rows[settings.group_row_by]])]
            
            #Sort the rows to make sure each subset is in alphabetical order by the sort column (order dpi so 14 dpi comes after 7 dpi)
            df_ordertemp.sort_values(by=[settings.matrix_row_order], inplace = True)
            
            data = df_temp.pivot_table(index=settings.matrix_row_names, 
                                columns=settings.matrix_col_names, 
                                values=settings.matrix_values,
                                aggfunc=settings.aggfunc)
            data = data.reindex(index = df_ordertemp[settings.matrix_row_names])
            
            tick_labels_list = ['']*len(data.columns)
            if settings.spacing_multiple is not None:
                print(settings.spacing_multiple)
                tick_labels_list = [str(i) if int(i) % settings.spacing_multiple <= settings.sliver_size else '' for i in list(data.columns)]
            #    tick_labels_list = list(range(min_x_axis,max_x_axis,spacing_multiple))
            if settings.first_last_label == 'first':
                tick_labels_list[0] =  str(min_value)
            if settings.first_last_label == 'last':
                tick_labels_list[len(tick_labels_list) - 1] =  str(max_value)
            if settings.first_last_label == 'first_and_last':
                tick_labels_list[len(tick_labels_list) - 1] =  str(max_value)
                tick_labels_list[0] =  str(min_value)
            if settings.first_last_label == 'all':
                tick_labels_list=  data.columns.map(str)

            
            
            #ticklabels = 
            if plot_rows * plot_cols == 1:
                axis2 = [axis]
            else:
                axis2 = axis.flat
            
            sns.heatmap(data, 
                        xticklabels=tick_labels_list, 
                        yticklabels=True,
                        ax=axis2[axis_num],
                        cbar=False, 
                        **heatmapkws)     
            #Remove the Axis and labels for Y in columns other than the first column 
            
            
            if n_col > 0:
                axis2[axis_num].tick_params(axis = 'y', 
                                                 which='both',      # both major and minor ticks are affected
                                                bottom=False,      # ticks along the bottom edge are off
                                                top=False,
                                                left=False, 
                                                right = False,# ticks along the top edge are off
                                                labelleft=False) # labels along the bottom edge are off)
            #Only include  the x-axis grouping labels in the top row ,and put the Group names on top
            if n_row == 0 and  col_labels:
                axis2[axis_num].xaxis.set_label_position('top') 
                axis2[axis_num].set_xlabel(col_rows[settings.group_col_by])
            else:
                axis2[axis_num].set_xlabel('')
            #Put the X scale at the bottom row, on the bottom only, row = last = bottom row
            if(n_row == plot_rows -1 ): 
                axis2[axis_num].tick_params(axis = 'x', labelrotation=75, width = 0)
            else:
                axis2[axis_num].tick_params(axis = 'x', 
                                             which='both',      # both major and minor ticks are affected
                                                bottom=False,      # ticks along the bottom edge are off
                                                top =False,         # ticks along the top edge are off
                                                labelbottom=False) # labels along the bottom edge are off)
                
            axis2[axis_num].set_ylabel('')
            axis2[axis_num].hlines(list(range(1,len(data.index))), *axis2[axis_num].get_xlim(),linewidth=1, color='w')
            
            n_col = n_col + 1
            axis_num = axis_num + 1
        n_row = n_row + 1
        
    cax = fig.add_axes([0.9,0.2,0.03,0.7])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    
    if settings.output_filepath:
        #plt.tight_layout()
        F = plt.gcf()
        for ext_i in settings.extension:
            F.savefig(settings.output_filepath + ext_i, dpi = (500))
            print("output file to:")
            print(settings.output_filepath + ext_i)
            plt.show()
    return