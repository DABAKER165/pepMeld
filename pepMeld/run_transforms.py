from .build_transforms import build_transforms


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
    transformations = build_transforms(args)
    transformations.execute_transformations()


    print("SCRIPT COMPLETED SEE OUT PUT FOR DETAILS!")
    print('Transformation Steps:')
    print(transformations.names)
    print('Skipped Transformation Steps:')
    print(transformations.skip)
