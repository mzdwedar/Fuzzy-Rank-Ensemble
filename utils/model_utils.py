import os
import tensorflow as tf
import pandas as pd

def save_model(model, model_name, history):
    ''' save the model and the metrics'''
    os.makedirs('saved_models', exist_ok=True)

    model_saved_name = model_name + ".h5"
    model.save("saved_models/" + model_saved_name)

    os.makedirs('csv_files', exist_ok=True)
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file =  "history_" + model_name + ".csv"
    filepath = "csv_files/" + hist_csv_file 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    print(f'{model_saved_name} saved')
    print(f'{hist_csv_file} saved')

def load_hdf5_model(model_name, path='saved_models'):
    '''load saved hdf5 model'''
    model_path = path + "/" + model_name + ".h5"
    model = tf.keras.models.load_model(model_path)

    print(f'{model_name} is loaded')

    return model