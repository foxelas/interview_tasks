import pandas as pd
import configparser
from time import strptime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#plt.ion()
from os.path import join

################# Read Settings #################

# The filename of the settings file
settings_filename = 'config.ini'


# Reads settings from file
def read_config():
    config = configparser.ConfigParser()
    config.read(settings_filename)
    # config.sections()
    return config


################# Initialize settings #################

# Converts a string from the settings file to a list of values
def get_list_from_setting_string(values):
    values_ = values.split(',')
    values_ = [x.strip() for x in values_]
    return values_


settings = read_config()
target_filename = settings['DEFAULT']['TargetFilename']

datetime_data_columns = get_list_from_setting_string(settings['DEFAULT']['DatetimeDataColumns'])
numeric_data_columns = get_list_from_setting_string(settings['DEFAULT']['NumericDataColumns'])
id_data_columns = get_list_from_setting_string(settings['DEFAULT']['IdColumns'])
categorical_data_columns = get_list_from_setting_string(settings['DEFAULT']['CategoricalDataColumns'])

excluded_variables_from_model_input = get_list_from_setting_string(
    settings['DEFAULT']['ExcludedVariablesFromModelInput'])
variable_to_predict = settings['DEFAULT']['VariableToPredict']

split_percentage = float(settings['RUN']['SplitPercentage'])
window_size = int(settings['RUN']['WindowSize'])


################# Read Input Data #################

# Reads data from a csv filename using pandas
def read_csv(filename):
    df = pd.read_csv(filename)
    return df


# Reads the specific input data based on the settings file
def read_input_data():
    print('Reading input data...\n')
    df = read_csv(target_filename)
    # print(df.to_string())
    return df


################# Assisting Functions for Columns #################

# Gets column names with numerical data
def get_number_columns(df):
    number_columns = df._get_numeric_data().columns.tolist()
    return number_columns


# Gets column names with categorical data
def get_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [x for x in categorical_columns if not x == 'timestamp']
    return categorical_columns


# Get basic statistics for numerical variables
def get_numerical_statistics(df):
    df_numerical_columns = df[numeric_data_columns]
    df_numerical_columns_stats = pd.concat([df_numerical_columns.min(axis=0, skipna=True),
                                            df_numerical_columns.max(axis=0, skipna=True),
                                            df_numerical_columns.mean(axis=0, skipna=True)],
                                           axis=1, keys=['min', 'max', 'mean'])
    print('**Statistics of numerical variables\n')
    print(df_numerical_columns_stats)
    print('\n')


# Get basic statistics for categorical variables
def get_categorical_statics(df):
    df_categorical_columns = df[categorical_data_columns]

    print('**Statistics of categorical variables\n')
    for column_name in df_categorical_columns.columns:
        print('Variable: ' + column_name)
        value_counts = df_categorical_columns[column_name].value_counts()
        total = df_categorical_columns[column_name].count()
        stats = pd.DataFrame(list(zip(value_counts.index, value_counts.values / total * 100)),
                             columns=['Category', 'Frequency (%)'])
        print(stats.to_string())

    print('\n')


# Encode categorical variables
def covert_categorical_to_numbers(df, method=None):
    if method is None:
        method = 'onehot'

    for column_name in categorical_data_columns:
        if method == 'onehot':
            onehot_df = pd.get_dummies(df[column_name], columns=[column_name], prefix=column_name)
            df.drop([column_name], axis=1, inplace=True)
            df = df.join(onehot_df)

        elif method == 'decimal':
            categories = df[column_name].unique()
            df[column_name].replace(categories, range(0, len(categories)), inplace=True)

        else:
            raise ValueError('Not supported method for the conversion of categorical data. Please select among ['
                             'onehot, decimal].\n')

    return df


# Encode datetime variables
def convert_timestamps_to_datetime(df):
    for column_name in datetime_data_columns:
        df[column_name] = [strptime(x, '%Y-%m-%d %H:%M:%S') for x in df[column_name]]

    return df


# Apply data scaling
def apply_data_scaling(df, scaling_type=None):
    number_columns = numeric_data_columns
    if scaling_type == 'minmax' or scaling_type is None:
        scaler = MinMaxScaler()
        df_numeric = df[number_columns]
        df_scaled = scaler.fit_transform(df_numeric).transpose()

        for (column_name, index) in zip(number_columns, range(0, len(number_columns))):
            df[column_name] = df_scaled[index]
        return df, scaler
    else:
        raise ValueError('Unsupported scaling type.')


# Gets data by location_id and value_type_id
def get_data_by_location_and_value_type(df, location_id, value_type_id):
    df_section = df.loc[(df['value_type_id'] == value_type_id) & (df['location_id'] == location_id)]
    return df_section


# Inspect the data by producing time plots
def inspect_data(df, location_id=None, value_type_id=None):
    plt.figure()
    if location_id is None:
        location_id = 116.0
    if value_type_id is None:
        value_type_id = 11

    df_section = get_data_by_location_and_value_type(df, location_id, value_type_id)
    plot_features = df_section['value']
    plot_features.index = df_section['timestamp']
    _ = plot_features.plot()
    plt.title("Values for Location_id=" + str(location_id) + ' and Value_type_id=' + str(value_type_id))
    save_path = join('report', 'images', 'vis_data_' + str(value_type_id) + '_' + str(location_id) + '.png')
    plt.savefig(save_path)
    #plt.show()


# Apply data preprocessing
def data_preprocessing(df):
    get_numerical_statistics(df)
    get_categorical_statics(df)
    inspect_data(df, 116.0, 11)
    inspect_data(df, 23.0, 14)
    inspect_data(df, 23.0, 16)

    df = covert_categorical_to_numbers(df)
    df = convert_timestamps_to_datetime(df)
    df, scaler = apply_data_scaling(df)

    print('**Example of preprocessed data:\n')
    print(df[:][0:5].to_string())

    print('\nPreprocessing finished.\n')
    return df, scaler


# Gets the input values
def get_input_values(df):
    return df[variable_to_predict].values
