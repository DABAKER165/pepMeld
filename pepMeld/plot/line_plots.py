class plot_line_settings:
    def __init__(self,
                 df = None,
                 title = None,
                 #percentile_lim = [10^-6,1-10^-6]  ,   
                 xlim = None,# [-5,10],
                 ylim = None,# [.9,1],
                 yticks = None,#[.1, 1,10,50,90,99,99.9],
                 xticks = None,#[-5,0,5],
                 value_col = 'INTENSITY',
                 x_col = 'POSITION',
                 grid_rows_column = None , # 'VIRUS'
                 grid_cols_column = None, # 'ANIMAL_ID'
                 legend_col = 'DPI_STRING',
                 legend_order = None,
                 output_filepath = None,
                 plot_type = 'line',
                 #probax = 'y',
                 ylabels = 'Intensity',
                 graph_size = 7
                ):
        self.df = df
        self.title =title
        #self.percentile_lim =percentile_lim
        self.xlim = xlim
        self.ylim = ylim
        self.yticks = yticks
        self.xticks = xticks
        self.value_col = value_col
        self.x_col = x_col
        self.output_filepath = output_filepath
        self.grid_rows_column = grid_rows_column
        self.grid_cols_column = grid_cols_column
        self.legend_col = legend_col
        self.legend_order = legend_order
        #self.probax = probax
        self.ylabels = ylabels
        self.graph_size = graph_size
        self.plot_type = plot_type

def plot_line_plot(settings):
    import numpy
    from matplotlib import pyplot as plt
    import seaborn
    import probscale
    if settings.plot_type == 'scatter':
        plot_type = plt.scatter
        marker_type = "."
    else:
        plot_type = plt.plot
        marker_type = ""
    keep_column_list = [settings.value_col, settings.x_col, settings.legend_col, settings.grid_rows_column, settings.grid_cols_column]
    #remove nones 
    keep_column_list = [x for x in keep_column_list if x is not None]
    print(keep_column_list)
    df2 = settings.df[keep_column_list]
    df2.dropna(inplace = True)
    df2.drop_duplicates(inplace = True)
    sort_list= []
    if settings.legend_col is not None:
        sort_list.append(settings.legend_col)        
    if settings.x_col is not None:
        sort_list.append(settings.x_col)
        
    if len(sort_list) > 0:
        df2.sort_values(by=sort_list,inplace=True)
   # print(df2)
    
    fg = (
        seaborn.FacetGrid(data=df2,  hue=settings.legend_col, row=settings.grid_rows_column, col=settings.grid_cols_column, margin_titles=True, size=settings.graph_size)
            #.set(xlim=settings.xlim, ylim=(0, 1)) #must set dummy values BEFORE the map step
            .map(plot_type, settings.x_col, settings.value_col, linewidth=.5,marker = marker_type)
            #.set( ylim=settings.ylim, xticks=settings.xticks, yticks=settings.yticks)
            .set_ylabels(settings.ylabels)
            .add_legend()

    )   
    if settings.output_filepath:
        fg.sa