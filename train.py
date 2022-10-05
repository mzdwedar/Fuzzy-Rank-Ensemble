import os
import argparse
from tensorflow.keras import optimizers
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.utils import np_utils

from utils.model import DenseNet, Inception, Xception
from utils.ensemble_utils import doFusion
from utils.data_pipeline import parse_function, encode_y
from utils.generate_dataset import get_training_dataset, kFold
from utils.model_eval import predict, compute_metrics
from utils.model_utils import save_model


def train(i, df, train_batch, val_batch, NUM_EPOCHS, num_classes, input_size):

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
                      .batch(train_batch)
                      .repeat()
                      .prefetch(buffer_size=AUTOTUNE)
                    )
    val_dataset = (val
                    .map(parse_function)
                    .batch(val_batch)
                    .prefetch(buffer_size=AUTOTUNE)
                  )

    steps_per_epoch = n_train // train_batch
    validation_steps = n_val // val_batch

    print()
    print('DenseNet-169:')
    print()

    model1 = DenseNet(input_size, num_classes)
    model1.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5), 
                    loss='sparse_categorical_crossentropy', metrics=['acc'])

    history1 = model1.fit(x = train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )
                         
    save_model(model1, "DenseNet-169", history1)

    print()
    print('InceptionV3:')
    print()

    model2 = Inception(input_size, num_classes)
    model2.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5),
                    loss='sparse_categorical_crossentropy', metrics=['acc'])

    history2 = model2.fit(x = train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    save_model(model2, "InceptionV3", history2)

    print()
    print('Xception:')
    print()

    model3 = Xception(input_size, num_classes)
    model3.compile(optimizer = optimizers.RMSprop(learning_rate=2e-5), 
                    loss='sparse_categorical_crossentropy', metrics=['acc'])

    history3 = model3.fit(x=train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    save_model(model3, "Xception", history3)

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
    parser.add_argument('--input_size', type=int, default=128, help='input size of the image')
    args = parser.parse_args()

    df = get_training_dataset(args.path)
    df = kFold(df)


    for i in range(1,5):
        train(i, df, args.batch_size, args.batch_size, args.num_epochs, args.num_classes, args.input_size)


