import tensorflow as tf

def load_hdf5_model(model_name, path='saved_models'):
    '''load saved hdf5 model'''
    model_path = path + "/" + model_name + ".h5"
    model = tf.keras.models.load_model(model_path)

    print(f'{model_name} is loaded')

    return model