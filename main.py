import part1 as pt1
import part2 as pt2


def build_and_evaluate_model_target_data(experiment, location_id, value_type_id, input_shape,
                                         activation, layers_number, neurons_number, loss, optimizer_name, epochs,
                                         batch_size, window_size, split_percentage):


    experiment_settings = dict(location_id=location_id, value_type_id=value_type_id, activation=activation,
                               layers_number=layers_number, neurons_number=neurons_number, loss=loss,
                               optimizer_name=optimizer_name, epochs=epochs, batch_size=batch_size,
                               window_size=window_size, split_percentage=split_percentage)
    pt2.write_experiment_settings(experiment, experiment_settings)


    df = pt1.read_input_data()
    df_preprocessed, scaler = pt1.data_preprocessing(df)

    df_section = pt1.get_data_by_location_and_value_type(df_preprocessed, location_id, value_type_id)

    input_values = pt1.get_input_values(df_section)
    input_values, target_values = pt2.apply_windows(input_values, window_size)

    input_values_train, target_values_train, input_values_test, target_values_test = pt2.split_train_test(input_values,
                                                                                                          target_values,
                                                                                                          split_percentage)

    trained_model = pt2.build_and_evaluate_model(experiment, input_values_train, target_values_train, input_values_test,
                                                 target_values_test, scaler, input_shape, activation, layers_number,
                                                 neurons_number, loss, optimizer_name, epochs, batch_size)

    print('Image results are saved in report\images\ ')
    return


################# Experiment 1 #################
#experiment1_settings = dict(location_id=116.0, value_type_id=11, activation='relu', layers_number=3,
#                            neurons_number=[32, 16, 8, 4], loss='mean_squared_error', optimizer_name='Adam', epochs=200,
#                            batch_size=32, window_size=pt1.window_size, split_percentage=pt1.split_percentage)


################# Experiment 2 #################
#experiment2_settings = dict(location_id=116.0, value_type_id=11, activation='sigmoid', layers_number=4,
#                            neurons_number=[32, 16, 8, 4, 4], loss='mean_squared_error', optimizer_name='SGD', epochs=200,
#                            batch_size=32, window_size=pt1.window_size, split_percentage=pt1.split_percentage)


