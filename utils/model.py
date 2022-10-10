from tensorflow.keras import layers, applications, models


def DenseNet(input_dims, num_classes):
  input_shape = (input_dims[0], input_dims[1], 3)
  densenet_model = applications.DenseNet169(input_shape=input_shape,
                                            include_top=False,
                                            weights="imagenet"
                                          )

  for layer in densenet_model.layers:
    layer.trainable = False

  x = layers.Flatten()(densenet_model.output)
  x = layers.Dense(1024, activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dense(num_classes, activation='softmax')(x)
  model = models.Model(densenet_model.input, x)
  
  return model

def Inception(input_dims, num_classes):
  input_shape = (input_dims[0], input_dims[1], 3)
  pre_trained_model2 = applications.InceptionV3(input_shape=input_shape,
                                                include_top = False,
                                                weights='imagenet')

  for layer in pre_trained_model2.layers:
    layer.trainable = False

  x = layers.Flatten()(pre_trained_model2.output)
  x = layers.Dense(1028,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(64,activation='relu')(x)
  x = layers.Dense(num_classes, activation='softmax')(x)
  model = models.Model(pre_trained_model2.input, x)
  
  return model

def Xception(input_dims, num_classes):
  input_shape = (input_dims[0], input_dims[1], 3)
  pre_trained_model = applications.Xception(input_shape = input_shape, 
                                            include_top=False,
                                            weights="imagenet")

  for layer in pre_trained_model.layers:
    layer.trainable = False

  x = layers.Flatten()(pre_trained_model.output)
  x = layers.Dense(256,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(32,activation='relu')(x)
  x = layers.Dense(num_classes, activation='softmax')(x)
  model1 = models.Model(pre_trained_model.input,x)
  
  return model1