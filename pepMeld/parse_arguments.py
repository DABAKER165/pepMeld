#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os

"""Parses the python script arguments from bash and makes sure files/inputs are valid"""


def parse_process_arrays_args(parser: ArgumentParser):
    """
        adds the arguments to the
    """
    parser.add_argument('--cores', type=int,
                        help='Number of cores to use')
    parser.add_argument('--log_base', type=int,
                        help='log (2 is default) to convert values')
    parser.add_argument('--input_path', type=str,
                        help='(required) default: None Raw data file ')
    parser.add_argument('--input_sep', type=str,
                        help='default: \\t seperator (i.e. , ; \t used to seporate values for input')
    parser.add_argument('--corr_key_path', type=str,
                        help='default: None')
    parser.add_argument('--corr_key_sep', type=str,
                        help='default: \\t seperator (i.e. , ; \\t used to seporate values for corresponding key')
    parser.add_argument('--corr_key_all_path', type=str,
                        help='(default: None seperator (i.e. , ; \\t used to seporate values for corresponding all key')
    parser.add_argument('--corr_key_all_sep', type=str,
                        help='default: \\t seperator (i.e. , ; \\t used to seporate values for corresponding all key')

    parser.add_argument('--corr_key_consolidated_iopath', type=str,
                        help='(default: None consolidated corresponding key, if you have a different from the nimblegen default')
    parser.add_argument('--corr_key_consolidated_sep', type=str,
                        help='default: \\t seperator (i.e. , ; \\t used to seporate values for corresponding all key')

    parser.add_argument('--metadata_path', type=str,
                        help='(required) Meta data file path, columns to convert the default column names to readable names and categories')
    parser.add_argument('--metadata_sep', type=str,
                        help='default: \\t')

    parser.add_argument('--virus_path', type=str,
                        help='Two column Sequence Name to Sequence Number')
    parser.add_argument('--virus_sep', type=str,
                        help='(optional) default: \\t')
    parser.add_argument('--protein_path', type=str,
                        help='(optional) Peptide Sequence to Protein Name (Can have additional uniquely named columns')
    parser.add_argument('--protein_sep', type=str,
                        help='default: \\t')

    parser.add_argument('--data_output_dir', type=str,
                        help='directory to output data files')
    parser.add_argument('--charts_output_dir', type=str,
                        help='directory to output data files')
    parser.add_argument('--output_sep', type=str,
                        help='delimiter for the output files'
                             'default: tab')
    parser.add_argument('--config', type=str,
                        help='filepath to arguments file that will override any set or default arguments it contains'
                             'in json format i.e. {"log_base":2, "input_sep":","}')
    parser.add_argument('--custom_columns', type=str,
                        help='filepath to custom column names, nested object (dict like)'
                             'in json format -- {filetype: {default_col: custom_col},...}'
                             'i.e. "input": {"PROBE_SEQ":"SEQUENCE", "INTENSITY":"VALUE"}')
    parser.add_argument('--version', action='version', version='pepMeld 0.1.0')


def check_required_fields(args):
    """Checks the require fields to make sure they are not blank"""
    require_inputs = ['transforms_path']
    for args_i in require_inputs:
        if args[args_i] is None:
            print('{0} must be declared'.format(args_i))
            raise NameError()


def check_seps(args):
    """Makes sure the separator is exactly one character (tab is one character)"""
    for args_i in args.keys():
        if args[args_i] is not None:
            if args_i[-4:] == '_sep':
                if len(args[args_i]) != 1:
                    print('separater must be one character: {0}'.format(args[args_i]))
                    raise NameError()
        else:
            # set default
            args[args_i] = '\t'


def check_paths(args):
    """Checks the paths using os.path, makes directories that don't exist"""
    import os
    for args_i in args.keys():
        if args[args_i] is not None:
            if args_i[-5:] == '_path':
                path_list = args[args_i].split(',')
                for path_i in path_list:
                    if not os.path.isfile(path_i):
                        print('file cannot be found for {0} : {1}'.format(args_i, args[args_i]))
                        raise NameError()
                args[args_i] = path_list
            if args_i[-10:] == 'output_dir':
                os.makedirs(args[args_i], exist_ok=True)
                print("Made directory: {0}".format(args[args_i]))


def get_process_arrays_args():
    """	Inputs arguments from bash
    Gets the arguments, checks requirements, returns a dictionary of arguments
    Return: args - Arguments as a dictionary
    """
    parser = ArgumentParser()
    parse_process_arrays_args(parser)
    args_ns = parser.parse_args()
    if args_ns.config is not None:
        print(args_ns.config)
        with open(args_ns.config) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args_ns, key, value)
    # convert namespace to an easier to use dictionary.
    args = {'cores': 1, 'log_base': 2}
    for k, v in args_ns._get_kwargs():
        # if v is not None:
        args[k] = v
    check_required_fields(args)

    check_paths(args)
    check_seps(args)
    # check_default_fields(args)
    return args
