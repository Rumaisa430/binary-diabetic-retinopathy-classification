
#Build the ResNet50 binary classification model.

import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model


def build_resnet50():

    inputs = Input(shape=(224, 224, 3))

    # Load pretrained ResNet50
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.4)(x)

    # Binary classification
    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

"""


import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model


def build_model():

    # Load pretrained ResNet50
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build classification model
    inputs = Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.4)(x)

    # Binary classification output
    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_resnet50(input_shape=(224, 224, 3), num_classes=1):
  # Load pre-trained ResNet50 base model without top classification layers
  base_model = ResNet50(
      weights="imagenet", include_top=False, input_shape=input_shape
  )

  # Keep base model layers trainable or fine-tuneable
  base_model.trainable = True

  # Add custom classification head
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation="relu")(x)

  # Binary classification uses 1 output unit with 'sigmoid' activation
  activation_func = "sigmoid" if num_classes == 1 else "softmax"
  outputs = Dense(num_classes, activation=activation_func)(x)

  model = Model(inputs=base_model.input, outputs=outputs)
  return model"""