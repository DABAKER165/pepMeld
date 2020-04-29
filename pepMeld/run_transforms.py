from .build_transforms import build_transforms
from .add_transformation_classes import execute_transformations
from .transformations.transform import df_files
from .transformations.open_files import *
import os
import pandas as pd


################################
#  Open Peptide Array Data Raw # 
################################
def import_consolidated(args):
    if args['corr_key_consolidated_iopath'] is None and args['corr_key_path'] is None and args[
        'corr_key_all_path'] is None:
        raise ('need at least one corr key')
    if args['corr_key_consolidated_iopath'] is None:
        return True
    if not os.path.isfile(args['corr_key_consolidated_iopath']):
        return True
    if args['corr_key_path'] is None or args['corr_key_all_path'] is None:
        return False
    if os.path.isfile(args['corr_key_path']) and os.path.isfile(args['corr_key_all_path']):
        return True
    return False


def run_transforms(args):
    data_class = df_files(file_type='data',
                          filepaths=args['input_path'],
                          meta_filepaths=args['metadata_path'],
                          data_sep=args['input_sep'],
                          meta_sep=args['metadata_sep'],
                          column_lookup={})
    if 'PEP_LEN' in data_class.df_dict['df'].columns:
        data_class.df_dict['df'] = data_class.df_dict['df'].loc[data_class.df_dict['df']['PEP_LEN'] != '5MER']

    ################################
    #  Get The Corresponding Data  #
    ################################
    # Import from multiple Corr files

    raw_corr_import = import_consolidated(args)
    ##
    if raw_corr_import:
        corr_class = corr_files(filepaths=args['corr_key_path'],
                                key_all_sep=args['corr_key_sep'])
        df_corr_merged = corr_class.df
        if os.path.isfile(args['corr_key_all_path']):
            corr_all_class = corr_all_files(filepaths=args['corr_key_all_path'], key_all_sep=args['corr_key_all_sep'])
            df_corr_merged = pd.concat([corr_class.df, corr_all_class.df], ignore_index=True)
            df_corr_merged = df_corr_merged.loc[df_corr_merged['SEQ_ID'] != 'REDUNDANT']
        if args['virus_path'] is not None and os.path.isfile(args['virus_path']):
            virus_class = virus_lookup_files(filepaths=args['virus_path'],
                                             key_all_sep=args['virus_sep'])

            on_columns = list(set(df_corr_merged.columns).intersection(set(virus_class.df.columns)))
            df_corr_merged = df_corr_merged.merge(virus_class.df,
                                                  how='inner',
                                                  on=on_columns)
            print('Inner Merged Corresponding Sequences and data (df_corr_merged, virus_class.df) on: ' + '\n'.join(
                on_columns))
        if args['protein_path'] is not None and os.path.isfile(args['protein_path']):
            protein_class = protein_lookup_files(filepaths=args['protein_path'],
                                                 default_class=default_protein_lookup_file(),
                                                 key_all_sep=args['protein_sep'])

            on_columns = list(set(df_corr_merged.columns).intersection(set(protein_class.df.columns)))
            df_corr_merged = df_corr_merged.merge(protein_class.df,
                                                  how='outer',
                                                  on=on_columns)

            print('Outer Merged Corresponding Sequences and data (df_corr_merged, protein_class.df) on: ' + '\n'.join(
                on_columns))
        if args['corr_key_consolidated_iopath'] is not None and os.path.isfile(args['corr_key_consolidated_iopath']):
            df_corr_merged.to_csv(args['corr_key_consolidated_iopath'], index=False, sep=args['corr_key_consolidated_sep'])
            print('outputed df_corr_merged to: ' + corr_key_outpath)
    # use an existing file that is ready for import
    else:
        print('Using consolidated Corr Key')
        df_corr_merged = pd.read_csv(args['corr_key_consolidated_iopath'], sep=args['corr_key_consolidated_sep'])
        df_corr_merged_rename_dict = {}
        df_corr_merged.rename(columns=df_corr_merged_rename_dict, inplace=True)

    data_class.add_dataframe(df_corr_merged, 'corr_merged')

    transformations = build_transforms(args)
    execute_transformations(data_class, transformations)
    # Optional
    # data_class.df_dict['merged_stacked_meta'].dropna(axis=1, how='all', inplace = True)

    # shutil.copy2(notebook_path, out_dir + '/' + EXPERIMENT + '.ipynb')

    # get_notebook_path(OUTPUT_FOLDER, main_dir, notebook_name, experiment)
    # copy_file_to_results(OUTPUT_FOLDER, data_meta_filepathlist)

    print("SCRIPT COMPLETED SEE OUT PUT FOR DETAILS!")
    print('Transformation Steps:')
    print(transformations.names)
    print('Skipped Transformation Steps:')
    print(transformations.skip)
