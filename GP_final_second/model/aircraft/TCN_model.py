import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam


def build_tcn_model(input_shape, num_filters=64, kernel_size=3, dilation_rate=1, num_layers=4):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_layers):
        x = Conv1D(filters=num_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate ** i,
                   padding='causal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Flatten the output to feed into Dense layers
    x = Lambda(lambda k: tf.reduce_mean(k, axis=1))(x)

    outputs = Dense(3, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model


# # Example usage
# input_shape = (100, 1)  # (time_steps, features)
# model = build_tcn_model(input_shape)
# model.summary()