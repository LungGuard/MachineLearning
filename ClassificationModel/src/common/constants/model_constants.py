


class ModelConstants:
    KERNEL_INITIALIZER='he_normal'
    RELU_ACTIVATION_FUNCTION='relu'
    OUTPUT_ACTIVATION = 'softmax'
    XAVIER_INITIALIZER = 'glorot_uniform'
    OUTPUT_LAYER_NAME='output'
    BN_PREFIX = 'bn'
    DROPOUT_PREFIX = 'dropout'
    DROPOUT_DENSE_PREFIX = 'dropout_dense'
    BN_DENSE_PREFIX = 'bn_dense'
    DENSE_PREFIX = 'dense'
    FLATTEN_NAME = 'flatten'
    PADDING_SAME = 'same'
    LOSS_CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    LOSS_METRIC = 'loss'

    METRIC_ACCURACY = 'accuracy'
    METRIC_PRECISION = 'precision'
    METRIC_RECALL = 'recall'
    BLOCK_PREFIXES = {
        1: 'conv1_',
        2: 'conv2_',
        3: 'conv3_',
        4: 'conv4_'
    }
    MAXPOOL_NAMES = {
        1: 'maxpool1',
        2: 'maxpool2',
        3: 'maxpool3',
        4: 'maxpool4'
    }
    EPOCHS = 100

