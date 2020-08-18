class plot_cdf_settings:
    def __init__(self,
                 df = None,
                 title = None,
                 percentile_lim = [10^-6,1-10^-6]  ,   
                 xlim = None,# [-5,10],
                 ylim = None,# [.9,1],
                 yticks = [.1, 1,10,50,90,99,99.9],
                 xticks = [-5,0,5],
                 value_col = 'INTENSITY',                 
                 grid_rows_column = None , # 'VIRUS'
                 grid_cols_column = None, # 'ANIMAL_ID'
                 legend_col = 'DPI_STRING',
                 legend_order = None,
                 output_filepath = None,
                 probax = 'y',
                 ylabels = 'Percentile',
                 graph_size = 5
                ):
        self.df = df
        self.title =title
        self.percentile_lim =percentile_lim
        self.xlim = xlim
        self.ylim = ylim
        self.yticks = yticks
        self.xticks = xticks
        self.value_col = value_col
        self.output_filepath = output_filepath
        self.grid_rows_column = grid_rows_column
        self.grid_cols_column = grid_cols_column
        self.legend_col = legend_col
        self.legend_order = legend_order
        self.probax = probax
        self.ylabels = ylabels
        self.graph_size = graph_size
        
def plot_cdf(settings):
    import numpy
    from matplotlib import pyplot
    import seaborn
    import probscale
    
    keep_column_list = [settings.value_col, settings.legend_col, settings.grid_rows_column, settings.grid_cols_column]
    #remove nones 
    keep_column_list = [x for x in keep_column_list if x is not None]
    print(keep_column_list)
    df2 = settings.df[keep_column_list]
    df2.dropna(inplace = True)
    df2.drop_duplicates(inplace = True)
 
    fg = (
        seaborn.FacetGrid(data=df2, hue=settings.legend_col, row=settings.grid_rows_column, col=settings.grid_cols_column, margin_titles=True, size=settings.graph_size)
            .set(xlim=settings.xlim, ylim=(0, 1)) #must set dummy values BEFORE the map step
            .map(probscale.probplot, settings.value_col, probax=settings.probax)
            .set( ylim=settings.ylim, xticks=settings.xticks, yticks=settings.yticks)
            .set_ylabels(settings.ylabels)
            .add_legend()

    )   
    if settings.output_filepath:
        fg.savefig(settings.output_filepath)
    return

