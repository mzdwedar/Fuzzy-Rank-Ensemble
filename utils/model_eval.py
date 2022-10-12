import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K


metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
                tfa.metrics.F1Score(num_classes=6, threshold=None, average='macro', name = 'f1_score'),
                ]

def compute_metrics(model_name, Y, preds):
  '''
  compute accuracy, specificity, recall, F1 Score, and AUC

  args
    model_name
    Y: grond truth labels
    preds: Numpy array(s) of predictions.
  '''

  cat_acc = tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy')
  f1score = tfa.metrics.F1Score(num_classes=6, threshold=None, average='macro', name = 'f1_score')
  recall = tf.keras.metrics.Recall(name='sensitivity/recall')
  auc = tf.keras.metrics.AUC(curve='ROC', name='AUC', multi_label=True, num_labels=6)
  specificity = tf.keras.metrics.SpecificityAtSensitivity(0.5)

  y_preds=[]

  for pred in preds: 
    y_preds.append(np.argmax(pred))

  print(f'{model_name} Accuracy Score: ',cat_acc(Y, y_preds).numpy())
  print(f'{model_name} specificity Score: ', specificity(Y, y_preds))
  print(f'{model_name} Recall Score: ', recall(Y, y_preds).numpy())
  print(f'{model_name} F1 Score: ',f1score(Y, y_preds).numpy()) 
  print(f'{model_name} AUC: ',auc(Y, y_preds).numpy()) 

def predict(model, dataset, num_examples, batch_size):
  '''
  Generates output predictions for the input.

  args
    model: tf.keras.model
    dataset: tf.data (X, y)
    num_examples: size of the dataset
    batch_size: an argument for model.predict
  
  returns
    Numpy array(s) of predictions.
  '''
  ds = dataset.unbatch()
  ds = ds.batch(num_examples)  
  
  X = []
  for images, _ in ds.take(1):
    X = images.numpy()

  preds = model.predict(X, batch_size=batch_size)
  
  return preds