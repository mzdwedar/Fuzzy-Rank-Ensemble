import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K



def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
                tfa.metrics.F1Score(num_classes=6,threshold=None,average='macro',name = 'f1_score'),
                tf.keras.metrics.Recall(name='sensitivity/recall'),
                specificity,
                tf.keras.metrics.AUC(curve='ROC',name='AUC',multi_label=True,num_labels=6)]

