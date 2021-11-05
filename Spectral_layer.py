from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.util.tf_export import keras_export
from tensorflow import multiply as mul
from tensorflow.python.keras.layers.ops import core as core_ops


@keras_export('keras.layers.Spectral')
class Spectral(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 is_base_trainable=True,
                 is_diag_trainable=True,
                 use_bias=False,
                 base_initializer='GlorotUniform', # 'optimized_uniform',
                 diag_initializer='optimized_uniform',
                 bias_initializer='zeros',
                 base_regularizer=None,
                 diag_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 base_constraint=None,
                 diag_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Spectral, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)

        self.is_base_trainable = is_base_trainable
        self.is_diag_trainable = is_diag_trainable
        self.use_bias = use_bias

        # 'optimized_uniform' initializers optmized by Buffoni and Giambagli
        if base_initializer == 'optimized_uniform':
            self.base_initializer = initializers.RandomUniform(-0.02, 0.02)
        else:
            self.base_initializer = initializers.get(base_initializer)
        if diag_initializer == 'optimized_uniform':
            # self.diag_initializer = initializers.RandomUniform(-0.5, 0.5)
            self.diag_initializer = initializers.RandomUniform(-0.5, 0.5)
        else:
            self.diag_initializer = initializers.get(diag_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.base_regularizer = regularizers.get(base_regularizer)
        self.diag_regularizer = regularizers.get(diag_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.base_constraint = constraints.get(base_constraint)
        self.diag_constraint = constraints.get(diag_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        # trainable eigenvector elements matrix
        # \phi_ij
        self.base = self.add_weight(
            name='base',
            shape=(input_shape[-1], self.units),
            initializer=self.base_initializer,
            regularizer=self.base_regularizer,
            constraint=self.base_constraint,
            dtype=self.dtype,
            trainable=self.is_base_trainable
        )

        # trainable eigenvalues
        # \lambda_i
        self.diag_end = self.add_weight(
            name='diag_end',
            shape=(1, self.units),
            initializer=self.diag_initializer,
            regularizer=self.diag_regularizer,
            constraint=self.diag_constraint,
            dtype=self.dtype,
            trainable=self.is_diag_trainable
        )

        # \lambda_j
        self.diag_start = self.add_weight(
            name='diag_start',
            shape=(input_shape[-1], 1),
            initializer=self.diag_initializer,
            regularizer=self.diag_regularizer,
            constraint=self.diag_constraint,
            dtype=self.dtype,
            trainable=self.is_diag_trainable
        )

        # bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, **kwargs):
        return core_ops.dense(
            inputs,
            mul(self.base, self.diag_start - self.diag_end),
            self.bias,
            self.activation,
            dtype=self._compute_dtype_object)
