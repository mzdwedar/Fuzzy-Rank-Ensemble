from gc import callbacks
import os
import argparse
from tensorflow.keras import optimizers, losses
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.utils import np_utils
from utils.metrics import *
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from utils.model import DenseNet, Inception, Xception
from utils.ensemble_utils import doFusion
from utils.data_pipeline import INPUT_DIMS, augment, normalize, parse_function, encode_y
from utils.generate_dataset import get_training_dataset, kFold
from utils.model_eval import predict, compute_metrics


def train(i, df, train_batch, val_batch, NUM_EPOCHS, num_classes, input_size, metrics_list):

    print(f"---------------------------------------FOLD NO {i}----------------------------------")
    
    dfTrain = df[df['kfold']!=i]
    dfVal = df[(df['kfold']==i)]
 
    y_true_val = dfVal['class'].apply(encode_y).to_numpy()

    
    n_val = len(dfVal)
    n_train = len(dfTrain)

    cols = ['path', 'class']
    train = tf.data.Dataset.from_tensor_slices(dfTrain[cols].to_numpy())
    val = tf.data.Dataset.from_tensor_slices(dfVal[cols].to_numpy())

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (train
                      .cache()
                      .shuffle(buffer_size=n_train)
                      .map(parse_function, num_parallel_calls=AUTOTUNE)
                      .map(augment, num_parallel_calls=AUTOTUNE)
                      .map(normalize, num_parallel_calls=AUTOTUNE)
                      .batch(train_batch)
                      .repeat()
                      .prefetch(buffer_size=AUTOTUNE)
                    )
    val_dataset = (val
                    .map(parse_function, num_parallel_calls=AUTOTUNE)
                    .map(normalize, num_parallel_calls=AUTOTUNE)
                    .batch(val_batch)
                    .prefetch(buffer_size=AUTOTUNE)
                  )

    steps_per_epoch = n_train // train_batch
    validation_steps = n_val // val_batch

    earlystopping = EarlyStopping(
      monitor='val_loss', patience=5, verbose=0, mode='auto',
      baseline=None, restore_best_weights=False) 

    print()
    print('DenseNet-169:')
    print()

    model1_checkpoint = ModelCheckpoint(filepath='saved_models/'+'DenseNet-169',
                                        monitor='val_cat_accuracy',
                                        save_best_only = True,
                                        save_weights_only  = True)
    model1_logger = CSVLogger('DenseNet-169.log')

    model1 = DenseNet(input_size, num_classes)
    model1.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5), 
                    loss=losses.SparseCategoricalCrossentropy(), 
                    metrics=metrics_list, callbacks=[model1_logger, earlystopping, model1_checkpoint])

    model1.fit(x = train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    print()
    print('InceptionV3:')
    print()

    model2_checkpoint = ModelCheckpoint(filepath='saved_models/'+'InceptionV3',
                                        monitor='val_cat_accuracy',
                                        save_best_only = True,
                                        save_weights_only  = True)
    model2_logger = CSVLogger('InceptionV3.log')

    model2 = Inception(input_size, num_classes)    
    model2.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5),
                    loss=losses.SparseCategoricalCrossentropy(), 
                    metrics=metrics_list, callbacks=[model2_logger, earlystopping, model2_checkpoint])

    model2.fit(x = train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    print()
    print('Xception:')
    print()

    model3_checkpoint = ModelCheckpoint(filepath='saved_models/'+'Xception',
                                        monitor='val_cat_accuracy',
                                        save_best_only = True,
                                        save_weights_only  = True)
    model3_logger = CSVLogger('Xception.log')

    model3 = Xception(input_size, num_classes)
    model3.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5), 
                    loss=losses.SparseCategoricalCrossentropy(), 
                    metrics=metrics_list, callbacks=[model3_logger, earlystopping, model3_checkpoint])

    model3.fit(x=train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    print("BASE LEARNERS ACCURACY-----------1.DENSENET 2.INCEPTION 3.XCEPTION")

    preds1 = predict(model1, val_dataset, n_val, val_batch)
    preds2 = predict(model2, val_dataset, n_val, val_batch)
    preds3 = predict(model3, val_dataset, n_val, val_batch)

    compute_metrics("DenseNet-169", y_true_val, preds1)
    compute_metrics("InceptionV3", y_true_val, preds2)
    compute_metrics("Xception", y_true_val, preds3)

    y_OHE = np_utils.to_categorical(y_true_val) # one hot encoded
    ensem_preds = doFusion(preds1, preds2, preds3, y_OHE, num_classes)

    print('Ensembled:')

    print(classification_report(y_true_val, ensem_preds, digits=4))

    print(f"--------------------------------------------------END OF FOLD NO {i}--------------------------------------------------------")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='Directory where the image data is stored')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for train and val')
    parser.add_argument('--num_classes', type=int, default=6, help='number of network outputs')
    args = parser.parse_args()

    os.makedirs('saved_models', exist_ok=True)
    df = get_training_dataset(args.path)
    df = kFold(df)    

    for i in range(1,5):
        train(i, df, args.batch_size, args.batch_size, args.num_epochs, args.num_classes, INPUT_DIMS, metrics_list)


