import argparse
import re

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("compare_lists")

target_column = "name"
brand_column = "brand"

# Encoding of brand names
brand_dictionary = {
    'corona extra': 'corona',  # TODO: add more
}

# Words to remove from brand names
brand_replace_list = ['brewing', 'brewery', 'company']

# Special characters to replace
special_characters = {'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}

# Wine types to find in product names
wine_types = ['pinot', 'sauvignon', 'chardonnay', 'merlot', 'getariako', 'cabernet', 'riesling', 'malbec', 'brut',
              'chenin', 'zinfandel', 'shiraz', 'sangiovese', 'prosecco', 'champagne', 'torrontes', 'fiano',
              'friulano', 'albarino', 'aransat', 'veltliner', 'muscadet', 'chablis', 'valmur', 'organic', 'vinho verde',
              'sancerre', 'vermentino', 'vouvray', 'picpoul', 'falanghina', 'orange', 'aligot', 'assyrtiko',
              'gavi', 'verdejo', 'moschofilero', 'verdicchio', 'california', 'moscato', 'quincy', 'blanc', 'rose',
              'white', 'ros', ]

# Packaging types to find in product names
packaging_delims = ['12pk', '6pk', '4pk', '12oz', '12fl', '12fz', '6 pack', '12 pack', '16fz', '4pack', '19.2fz',
                    'variety pack', '19.2oz can', '25oz']


def clean_text(text):
    if not pd.notna(text):
        return ''

    text = text.lower()  # convert to lowercase
    text = text.strip()  # delete spaces at the beginning and end of the string

    for letter, rep in special_characters.items():  # replace special characters
        text = text.replace(letter, rep)
    text = re.sub(r'[^a-zA-Z0-9,\s]', '', text)     # Keep only alphanumeric characters
    text = text.replace('  ', ' ')  # remove double spaces
    return text


def clean_brand(text):
    text = clean_text(text)
    if text in brand_dictionary.keys():  # replace brand name with encoding
        text = brand_dictionary[text]
    for pattern in brand_replace_list:  # remove words from brand name
        text = re.sub(pattern, '', text)

    if len(text) > 3 and ' co' in text:  # remove ' co' from the end of the string
        if text[-3:] == ' co':
            text = text[:-3]
    return text.strip()


def clean_value(target, brand):
    target = clean_text(str(target))
    brand = clean_text(str(brand))
    parts = target.split(",")
    name = parts[0]
    name = name.replace(brand, "", 1)

    if len(parts) > 1:  # strip brand from product name
        package = parts[-1]
    else:
        package = None

    for delim in packaging_delims:
        if delim in name:
            parts = name.split(delim)
            package = delim + delim.join(parts[1:])
            name = parts[0]
            break

    if brand == '':  # if brand is empty, it's probably the first word in the name (especially for wine)
        for delim in wine_types:
            if delim in name:
                parts = name.split(delim)
                name = delim + (delim.join(parts[1:]) if len(parts) > 1 else '')
                brand = parts[0]
                break

    name = clean_text(name)
    name = re.sub(r'\d', '', name).strip()  # remove numbers and spaces
    brand = clean_brand(brand)
    if package is not None:
        package = clean_text(package)
    return [name, package, brand]


def compare_against(df, target, target_brand):
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
    df[brand_column] = [clean_text(x) for x in df[brand_column].values]
    zipped_vals = [clean_value(x, y) for (x, y) in zip(df[target_column], df[brand_column])]
    df = pd.DataFrame(zipped_vals, columns=[target_column, 'package', brand_column])
    df.sort_values(by=[brand_column, target_column], inplace=True)
    return df


def compare_lists(input_file1, input_file2, output_file=None):
    if '.csv' not in input_file1 or '.csv' not in input_file2:
        logger.info("Please input valid .csv filenames.")

    if not output_file:
        output_file = "output.csv"

    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)
    logger.debug(f"Original List1:\n{df1.head()}\n")
    logger.debug(f"Original List2:\n{df2.head()}\n")

    df1 = preprocess_df(df1)
    df2 = preprocess_df(df2)

    logger.debug(f"Preprocessed List1:\n{df1.head()}\n")
    logger.debug(f"Preprocessed List2:\n{df2.head()}\n")

    #df1.to_csv("df1.csv")
    #df2.to_csv("df2.csv")

    comp = [(y, x) for (x, y) in zip(df1[target_column], df1[brand_column]) if compare_against(df2, x, y)]
    comp_df = pd.DataFrame(comp, columns=[brand_column, target_column])
    comp_df.drop_duplicates(inplace=True)
    comp_df.reset_index(drop=True, inplace=True)

    logger.debug(f"Result:\n{comp_df.head()}\n")
    comp_df.to_csv(output_file)

    logger.info(f"A total of {len(comp_df)} common items were found. The result is saved in {output_file}.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find common items between two CSV files.")
    parser.add_argument('input1', type=str, help='the filename of the first file')
    parser.add_argument('input2', type=str, help='the filename of the second file')
    parser.add_argument('output', type=str, help='the filename of the output file')

    argv = ["fresh_direct.csv", "whole_foods.csv", "output.csv"]
    opt = parser.parse_args(argv)
    compare_lists(opt.input1, opt.input2, opt.output)
