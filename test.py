import argparse

from utils.generate_dataset import get_testing_dataset
from utils.ensemble_utils import doFusion
from utils.model_utils import load_hdf5_model
from utils.data_pipeline import *
from utils.model_eval import predict
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf



def test(test_paths, batch_size):
    '''
    1. build data pipeline
    2. load the models
    3. pass the dataset to the models
    4. run fuzzy distance
    5. compute the metrics
    '''
    num_examples_test = len(test_paths)
    y_true = test_paths['class'].apply(encode_y).to_numpy()
    
    test = tf.data.Dataset.from_tensor_slices(test_paths.values)

    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = (test
                      .map(parse_function, num_parallel_calls=AUTOTUNE)
                      .batch(batch_size)
                      .prefetch(buffer_size=AUTOTUNE)
                    )

    model1 = load_hdf5_model("DenseNet-169")
    model2 = load_hdf5_model("InceptionV3")
    model3 = load_hdf5_model("Xception")

    preds1 = predict(model1, test_dataset, num_examples_test, batch_size)
    preds2 = predict(model2, test_dataset, num_examples_test, batch_size)
    preds3 = predict(model3, test_dataset, num_examples_test, batch_size)

    predictedClass = doFusion(preds1, preds2, preds3, y_true, num_examples_test)

    print('Ensembled')

    print('Accuracy score: ', accuracy_score(y_true, predictedClass))
    print(classification_report(y_true, predictedClass, digits=4))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
 
    parser.add_argument('--path', type=str, default='./',
                        help='Path where the image data is stored')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch Size for Mini Batch')
    
    args = parser.parse_args()

    test_df = get_testing_dataset(args.path)

    test(test_df, args.batch_size)