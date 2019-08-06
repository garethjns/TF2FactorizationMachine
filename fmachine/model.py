import tensorflow as tf


class FactorizationMachine:
    def __init__(self,
                 m: int = 2,
                 k: int = 5) -> None:
        """
        Initialise coefficients.

        For now:
         - Requires manual specification of feature space dimensionality.
         - Coefficients are all float64. Might be overkill but it simplifies casting.

        :param m: Number of features.
        :param k: Number of latent factors to model in V.
        """
        self.b = tf.Variable(tf.zeros([1],
                                      dtype='double'))
        self.w = tf.Variable(tf.random.normal(([m]),
                                              stddev=0.01,
                                              dtype='double'))
        self.v = tf.Variable(tf.random.normal(([k, m]),
                                              stddev=0.01,
                                              dtype='double'))

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Predict from model.

        :param x: Tensor containing features.
        :return: Tensor containing predictions.
        """
        # Linear terms
        linear = tf.reduce_sum(tf.multiply(self.w, x),
                               axis=1,
                               keepdims=True)

        # Interaction terms
        interactions = tf.multiply(0.5, tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(x, tf.transpose(self.v)), 2),
                                                                  tf.matmul(tf.pow(x, 2),
                                                                            tf.transpose(tf.pow(self.v, 2)))),
                                                      axis=1,
                                                      keepdims=True))

        # Linear sum along with intercept
        wv = tf.add(linear, interactions)
        bwv = tf.add(self.b, wv)

        return bwv
