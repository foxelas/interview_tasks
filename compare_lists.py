import argparse

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("compare_lists")

target_column = "name"
brand_column = "brand"

def clean_value(target, brand):
    target = str(target.lower())
    brand = str(brand.lower())
    # TODO: add more extensive cleanup and make encoding also
    parts = target.split(",")
    name = parts[0]
    name = name.replace("(", "").replace(")", "").replace(brand, "", 1)
    # strip brand from product name
    if len(parts) > 1:
        package = parts[-1]
    else:
        package = None
    target_delims = ['12pk', '6pk', '4pk', '12oz', '12fl', '12fz', '6 pack', '12 pack']
    for delim in target_delims:
        if delim in name:
            package = delim + name.split(delim)[1]
            name = name.split(delim)[0]
    return [name, package]


def compare_against(df, target, target_brand):
    if target_brand == 'nan':
        return False # TODO: handle Nan brands

    df_ = df[df[brand_column] == target_brand]
    if len(df_) > 0:
        result = [target for x in df_[target_column].values if target in x]
        result = len(result) > 0
        # TODO: also compare other columns apart from name and brand e.g. package type
        return result
    else:
        return False


def preprocess_df(df):
    df = df[[brand_column, target_column]].copy()
    df[brand_column] = [str(x).lower() if str(x) != "NaN" else '' for x in df[brand_column].values]
    zipped_vals = [clean_value(x, y) for (x,y) in zip(df[target_column], df[brand_column])]
    df[target_column] = [x[0] for x in zipped_vals]
    df["package"] = [x[1] for x in zipped_vals]
    return df


def compare_lists(input_file1, input_file2, output_file=None):
    if '.csv' not in input_file1 or '.csv' not in input_file2:
        logger.info("Please input valid .csv filenames.")

    if not output_file:
        output_file = "output.csv"

    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)
    logger.debug(f"List1:\n{df1.head()}")
    logger.debug(f"List2:\n{df2.head()}")

    df1 = preprocess_df(df1)
    df2 = preprocess_df(df2)

    logger.debug(f"List1:\n{df1.head()}")
    logger.debug(f"List2:\n{df2.head()}")

    comp = [y + x for (x,y) in zip(df1[target_column], df1[brand_column]) if compare_against(df2, x, y)]
    comp = np.unique(comp)

    comp_df = pd.DataFrame({target_column: comp})
    logger.debug(f"Result:\n{comp_df.head()}")
    comp_df.to_csv(output_file)

    logger.info(f"The result is saved in {output_file}.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input1', type=str, default='fresh_direct.csv', help='the filename of the first file')
    parser.add_argument('input2', type=str, default='wholefoods.csv', help='the filename of the second file')
    parser.add_argument('output', type=str, default='output.csv', help='the filename of the output file')
    opt = parser.parse_args()

    compare_lists(opt.input1, opt.input2, opt.output)


