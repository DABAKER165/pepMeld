#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os

"""Parses the python script arguments from bash and makes sure files/inputs are valid"""


def parse_process_arrays_args(parser: ArgumentParser):
    """
        adds the arguments to the
    """
    parser.add_argument('--transforms_path', type=str,
                        help='filepath to arguments file that will override any set or default arguments it contains'
                             'in json format i.e. {"log_base":2, "input_sep":","}')

    parser.add_argument('--version', action='version', version='pepMeld 0.1.0')


def check_required_fields(args):
    """Checks the require fields to make sure they are not blank"""
    require_inputs = ['transforms_path']
    for args_i in require_inputs:
        if args[args_i] is None:
            print('{0} must be declared'.format(args_i))
            raise NameError()





def get_process_arrays_args():
    """	Inputs arguments from bash
    Gets the arguments, checks requirements, returns a dictionary of arguments
    Return: args - Arguments as a dictionary
    """
    parser = ArgumentParser()
    parse_process_arrays_args(parser)
    args_ns = parser.parse_args()
    # convert namespace to an easier to use dictionary.
    args = {}
    for k, v in args_ns._get_kwargs():
        # if v is not None:
        args[k] = v
    check_required_fields(args)

    # check_default_fields(args)
    return args
