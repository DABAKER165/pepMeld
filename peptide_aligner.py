import pandas as pd
import os
from Bio import SeqIO
import os
import subprocess

# !/usr/bin/env python
from argparse import ArgumentParser, ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def parse_process_arrays_args(parser: ArgumentParser):
    """Parses the python script arguments from bash and makes sure files/inputs are valid"""
    parser.add_argument('--fasta_path',
                        type=str,
                        help='filepath to the fasta. The fasta seq_ids must match ids in the  protein groups path',
                        required=True)
    parser.add_argument('--seq_path',
                        type=str,
                        default='t',
                        help='path of your stacked peptide sequences with a PROBE_SEQUENCE, SEQUENCE_ID and POSITION column.',
                        required=True)
    parser.add_argument('--out_dir',
                        type=str,
                        default='./out',
                        help='Directory files are outputted',
                        required=False)
    parser.add_argument('--prot_groups_path',
                        type=str,
                        help='Table of first column SEQ_NAME (could be a common name) Each column is a different protein , each entry is an entry from the FASTA HEADER',
                        required=True)
    parser.add_argument('--df_data_path',
                        type=str,
                        help='If you want to join peptide data directly to the table, enter the path of the stacked data as created by pepMeld or equivelent format.',
                        required=False)
    parser.add_argument('--seq_comp_id',
                        type=str,
                        help='Capital Sensitive, exact match to the SEQ_NAME column (first column) of the prot_groups_path file',
                        required=True)
    parser.add_argument('--peptide_length',
                        type=int,
                        default=16,
                        help='Set to the maximum Peptide length in your seq_path',
                        required=False)
    parser.add_argument('--comparing_strain_prefix',
                        type=str,
                        default="",
                        help='Uses the SEQ_NAME (first column) of prot_groups_path table if blank, prefiex assigned in column names for csv files',
                        required=False)
    parser.add_argument('--muscle_path',
                    type=str,
                    default="muscle",
                    help='path to muscle program, https://www.drive5.com/muscle',
                    required=False)


def get_process_arrays_args():
    """	Inputs arguments from bash
    Gets the arguments, checks requirements, returns a dictionary of arguments
    Return: args - Arguments as a dictionary
    """
    parser = ArgumentParser()
    parse_process_arrays_args(parser)
    args = parser.parse_args()
    return args


args = get_process_arrays_args()
# same arguments to a local variable by same name as the argument
fasta_path = args.fasta_path
seq_path = args.seq_path
out_dir = args.out_dir
prot_groups_path = args.prot_groups_path
df_data_path = args.df_data_path
seq_comp_id = args.seq_comp_id
peptide_length = args.peptide_length
comparing_strain_prefix = args.comparing_strain_prefix
muscle_path = args.muscle_path

# in_dir = "/content/drive/Shared drives/dholab/data/peptide_array/rhesus_covid19"
# out_dir = os.path.join(in_dir, "out", "alignment")
# # muscle executable path
# muscle_path = os.path.join(in_dir, "muscle")
# fasta_dir = os.path.join(in_dir, 'fasta_inputs')
# complete_fasta_path = os.path.join(fasta_dir, 'covid_except_wi.fasta')
# seq_path = os.path.join(in_dir, 'all_seq_except_wi.tsv.gz')
# df_data_path = os.path.join(in_dir, 'df_stacked.tsv.gz')
# prot_groups_path = os.path.join(out_dir, 'covid_spike_mapping_groups.csv')
#
# seq_comp_id = 'NC_045512.2;YP_009724390.1;Wu1-SARS2_surface'
# comparing_strain_prefix = 'WU1-SARS2'
# peptide_length = 16


def open_fasta_as_df(fasta_path, protein):
    """
    :param fasta_path (string): path of multiple strain fasta file with all of the strain sequences (in ammino acid form)
    :param protein (string): Which protein group it will denote for the CSV
    :return:
    """
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    with open(fasta_path) as fasta_file:  # Will close handle cleanly
        seq_id = []
        seq_prot = []
        seq = []
        for title, sequence in SimpleFastaParser(fasta_file):
            seq_id.append(title)  # First word is ID
            seq_prot.append(title.split(';')[0])
            seq.append(sequence)
    return pd.DataFrame({'SEQ_ID': seq_id, 'SEQ_PROT': seq_prot, 'SEQUENCE': seq, 'PROTEIN': protein})


def get_aligned_positions_df(df_fasta,
                             seq_comp_id,
                             comparing_strain_prefix="COMPARING",
                             peptide_length=16):
    import pandas as pd
    # Easier to work with dictionary than iterrate the rows of data frame
    seq_dict = df_fasta.set_index('SEQ_ID')['SEQUENCE'].to_dict()
    seq_comp = seq_dict[seq_comp_id]
    seq_aligned_positions = {}
    total_aligned_score = {}
    seq_alignment_score_list = {}
    df_all = pd.DataFrame()
    for seq_id, seq in seq_dict.items():
        alignment_score = []
        # First over arching position.  Tracks the position of the comparing sequence
        j = 1
        score = 0
        # uses this to track the comparing sequence position
        comp_position = []
        # loop through the length of the sequence
        # Because the file (should) be aligned they should be the same length
        for i in range(0, len(seq)):
            # add to j because it is not a -
            if seq_comp[i] != '-':
                j = j + 1

            if seq[i] != '-':
                # this is the position that is matched to the comparing seq
                comp_position.append(j)
                # add to the total socre if they are the same (already check for -)
                if seq[i] == seq_comp[i]:
                    score = score + 1
                # track the end of the comparing sequence as it loops to i + k
                k = i
                # used to track valid positions of the comparing sequence
                m = 0
                # used to track the valid positions of the sequece of interest in this loop
                p = 0
                align_score = 0
                # Seq length
                # loops to the end of the peptide.
                # p and m are tracked for non - (amino acids declared)
                while (p < peptide_length) and (m < peptide_length):
                    # Break if the comparing sequence is at the end.
                    # Nothing else to align
                    if k + 1 > len(seq_comp):
                        break
                    # Track the comparing sequence non -  value count
                    # vs peptide length
                    if seq_comp[k] != '-':
                        m = m + 1
                    # Track the sequence non -  value count
                    # vs peptide length
                    if seq[k] != '-':
                        p = p + 1
                    # If both are not - and match add one to the align score
                    if (seq_comp[k] != '-') and (seq[k] != '-'):
                        if seq_comp[k] == seq[k]:
                            align_score = align_score + 1

                    k = k + 1

                alignment_score.append(align_score)

        # add to the dictionaries to store the values by seq_Id
        # seq_aligned_positions is the position of the comparing sequence
        seq_aligned_positions[seq_id] = comp_position
        # total_aligned_score is the total match count (subtracting '-' for both)
        # useful in ordering the sequences by over all mismatch count
        total_aligned_score[seq_id] = score
        # this is a list of each alignment match count
        seq_alignment_score_list[seq_id] = alignment_score
        df = pd.DataFrame({'SEQ_ID': seq_id,
                           'POSITION': list(range(1, len(comp_position) + 1)),
                           '{0}_POSITION'.format(comparing_strain_prefix): comp_position,
                           'MATCH_COUNT': alignment_score,
                           'SCORE': score})
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


def open_fasta_as_df(fasta_path, protein):
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    with open(fasta_path) as fasta_file:  # Will close handle cleanly
        seq_id = []
        seq_prot = []
        seq = []
        for title, sequence in SimpleFastaParser(fasta_file):
            seq_id.append(title)  # First word is ID
            seq_prot.append(title.split(';')[0])
            seq.append(sequence)
    return pd.DataFrame({'SEQ_ID': seq_id, 'SEQ_PROT': seq_prot, 'SEQUENCE': seq, 'PROTEIN': protein})


def get_aligned_positions_df(df_fasta,
                             seq_comp_id,
                             comparing_strain_prefix="COMPARING",
                             peptide_length=16):
    import pandas as pd
    import math
    # Easier to work with dictionary than iterrate the rows of data frame
    seq_dict = df_fasta.set_index('SEQ_ID')['SEQUENCE'].to_dict()
    seq_comp = seq_dict[seq_comp_id]
    seq_aligned_positions = {}
    total_aligned_score = {}
    seq_alignment_score_list = {}
    df_all = pd.DataFrame()
    for seq_id, seq in seq_dict.items():
        if seq == '':
            continue
        alignment_score = []
        # First over arching position.  Tracks the position of the comparing sequence
        j = 1
        score = 0
        # uses this to track the comparing sequence position
        comp_position = []
        # loop through the length of the sequence
        # Because the file (should) be aligned they should be the same length
        for i in range(0, len(seq)):
            # add to j because it is not a -
            if seq_comp[i] != '-':
                j = j + 1

            if seq[i] != '-':
                # this is the position that is matched to the comparing seq
                comp_position.append(j)
                # add to the total socre if they are the same (already check for -)
                if seq[i] == seq_comp[i]:
                    score = score + 1
                # track the end of the comparing sequence as it loops to i + k
                k = i
                # used to track valid positions of the comparing sequence
                m = 0
                # used to track the valid positions of the sequece of interest in this loop
                p = 0
                align_score = 0
                # Seq length
                # loops to the end of the peptide.
                # p and m are tracked for non - (amino acids declared)
                while (p < peptide_length) and (m < peptide_length):
                    # Break if the comparing sequence is at the end.
                    # Nothing else to align
                    if k + 1 > len(seq_comp):
                        break
                    # Track the comparing sequence non -  value count
                    # vs peptide length
                    if seq_comp[k] != '-':
                        m = m + 1
                    # Track the sequence non -  value count
                    # vs peptide length
                    if seq[k] != '-':
                        p = p + 1
                    # If both are not - and match add one to the align score
                    if (seq_comp[k] != '-') and (seq[k] != '-'):
                        if seq_comp[k] == seq[k]:
                            align_score = align_score + 1

                    k = k + 1

                alignment_score.append(align_score)

        # add to the dictionaries to store the values by seq_Id
        # seq_aligned_positions is the position of the comparing sequence
        seq_aligned_positions[seq_id] = comp_position
        # total_aligned_score is the total match count (subtracting '-' for both)
        # useful in ordering the sequences by over all mismatch count
        total_aligned_score[seq_id] = score
        # this is a list of each alignment match count
        seq_alignment_score_list[seq_id] = alignment_score
        df = pd.DataFrame({'SEQ_ID': seq_id,
                           'POSITION': list(range(1, len(comp_position) + 1)),
                           '{0}_POSITION'.format(comparing_strain_prefix): comp_position,
                           'MATCH_COUNT': alignment_score,
                           'SCORE': score})
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


df_prot_groups = pd.read_csv(prot_groups_path, sep=',')
df_seq = pd.read_csv(seq_path, sep='\t')
df_seq = df_seq[['POSITION',
                 'PROBE_SEQUENCE',
                 'SEQ_ID']]
if df_data_path is not None:
    df_data = pd.read_csv(df_data_path, sep='\t')
    df_data.drop(columns=['CELL', 'VENDOR_NAME', 'EXCLUDE'], inplace=True, errors='ignore')
df_prot_groups['SEQ_NAME']
column_list = list(df_prot_groups.columns)
column_list.pop(0)
df_aligned_data_all = pd.DataFrame()
df_aligned_all = pd.DataFrame()
# Create Fasta subs
os.makedirs(out_dir, exist_ok=True)
fasta_dir = os.path.join(out_dir, 'intermediate')
os.makedirs(fasta_dir, exist_ok=True)
for column_i in column_list:

    print(column_i)
    protein = column_i
    header_list = list(df_prot_groups[column_i])

    out_path = os.path.join(fasta_dir, '{0}_aligned.fasta'.format(protein))
    in_path = os.path.join(fasta_dir, '{0}.fasta'.format(protein))
    # create a new fasta with only from teh columns
    f = open(in_path, "a")
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.description in header_list:
            f.write(record.format("fasta"))
    f.close()
    # Run on sub grouping.
    subprocess.call(['{0} -in {1} -out {2}'.format(muscle_path,
                                                   in_path,
                                                   out_path)],
                    shell=True)
    # Open the aligned fasta file as a dataframe,
    # Columns SEQ_ID, SEQ_PROT, SEQUENCE, PROTEIN
    df_fasta = open_fasta_as_df(out_path, protein)

    # Get the aligned positions to the compaing straing
    if comparing_strain_prefix == '':
        comparing_strain_prefix = seq_comp_id
    seq_to_head_dict = df_prot_groups.set_index('SEQ_NAME')[protein].to_dict()

    df_aligned = get_aligned_positions_df(df_fasta=df_fasta,
                                          seq_comp_id=seq_to_head_dict[seq_comp_id],
                                          comparing_strain_prefix=comparing_strain_prefix,
                                          peptide_length=peptide_length)

    # Add a prtein column so it is more descernable
    df_aligned['PROTEIN'] = column_i
    # Merge teh Probe Sequence on POSITION and SEQ_ID
    df_aligned = df_aligned.merge(df_seq,
                                  on=['POSITION', 'SEQ_ID'],
                                  how='inner')

    # Filter for and rename the columns to merge the comparing sequence Probe sequence
    df_seq_compare = df_seq[df_seq['SEQ_ID'] == seq_to_head_dict[seq_comp_id]][['POSITION',
                                                              'PROBE_SEQUENCE']]

    df_seq_compare.rename(columns={'PROBE_SEQUENCE': '{0}_PROBE_SEQUENCE'.format(comparing_strain_prefix),
                                   'POSITION': '{0}_POSITION'.format(comparing_strain_prefix)},
                          inplace=True)

    # merge the comparing PROBE_SEQUENCE to the set based on the compare position
    df_aligned = df_aligned.merge(df_seq_compare,
                                  on=['{0}_POSITION'.format(comparing_strain_prefix)],
                                  how='inner')

    df_aligned.to_csv(os.path.join(out_dir, '{0}_aligned_seq_only.csv'.format(protein)),
                                    index=False)
    # MERGE THE data
    if df_data_path is not None:
        df_aligned_data = df_data.merge(df_aligned,
                                        on=['PROBE_SEQUENCE'],
                                        how='inner')
        # Save the data as a csv by column name (protein) to make smaller files
        df_aligned_data.to_csv(os.path.join(out_dir, '{0}_aligned_intensity_stacked.csv'.format(protein)),
                               index=False)
        # concat to a master file
        df_aligned_data_all = pd.concat([df_aligned_data_all,
                                         df_aligned_data],
                                        ignore_index=True)
    # concat to a master file
    df_aligned_all = pd.concat([df_aligned_all,
                                df_aligned],
                               ignore_index=True)


df_aligned_all.to_csv(os.path.join(out_dir, 'all_aligned_seq_only.csv'),
                      index=False)
if df_data_path is not None:
    df_aligned_data_all.to_csv(os.path.join(out_dir, 'all_aligned_intensity_stacked.csv'),
                               index=False)
