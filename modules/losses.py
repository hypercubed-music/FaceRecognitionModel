import tensorflow as tf


def SoftmaxLoss(num_classes):
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist

        '''y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)'''
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_onehot = tf.one_hot(y_true, num_classes)
        #tf.print(y_pred)
        #tf.print(tf.reduce_max(y_pred))
        #tf.print(tf.reduce_min(y_pred))
        smax = tf.nn.softmax(y_pred)
        #tf.print(smax)
        #tf.print(tf.reduce_max(smax))
        loss = -tf.reduce_sum(y_true_onehot*tf.math.log(smax + 1e-10), axis=1)
        return tf.reduce_mean(loss)
    return softmax_loss
