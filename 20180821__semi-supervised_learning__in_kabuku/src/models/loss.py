import tensorflow as tf


def TF_nnPU_loss(logits, labels, positive_pct = 0.2, beta = 0., gamma = 1., nnPU = True):
    """
    logits : shape (batch, 1)
    labels : shape (batch, )

    ex :
        logits = [[0],
                  [1],
                  [0]]
        labels = [P,U,P]
    positive_pct : 0 < positive_pct < 1
        positive and negative percentage is known beforehand

    reference :
        - Non-negative Positive-Unlabeled (nnPU) and unbiased Positive-Unlabeled (uPU) learning
          reproductive code on MNIST and CIFAR10
        - https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py
    """
    assert (positive_pct > 0) and (positive_pct < 1)
    positive_label = 1
    unlabeled_label = -1
    #________________________________________________________________________________
    ##      
    loss_func = lambda x: tf.sigmoid(x=-x)  # sigmoid: y = 1 / (1 + exp(-x))

    logits  = tf.reshape(tensor=logits, shape=[-1])
    y_label = tf.cast(x=tf.reshape(tensor=labels, shape=[-1]), dtype=logits.dtype)
    print(logits.get_shape())
    print(y_label.shape)
    #assert logits.shape == y_label.shape
    logits.get_shape().assert_is_compatible_with(other=y_label.get_shape())
    
    bool_positive, bool_unlabeled = tf.equal(x=y_label, y=positive_label), tf.equal(x=y_label, y=unlabeled_label)
    print(bool_positive.dtype)
    print(tf.cast(bool_positive, dtype=logits.dtype))

    ##  count positive and unlabel (Only Support |P| + |U| = |X| or |U| = |X| )
    ##  y_label has 0 (Unlabel) or 1 (Positive)
    n_positive  = tf.maximum(x=tf.reduce_sum(input_tensor=tf.cast(x=bool_positive, dtype=tf.float32)),
                             y=positive_label)
    #  n_unlabeled =  "batch" - n_positive
    n_unlabeled = tf.cast(x=tf.maximum(x=tf.reduce_sum(input_tensor=y_label), y=unlabeled_label), dtype=tf.float32)
    n_unlabeled  = tf.maximum(x=tf.reduce_sum(input_tensor=tf.cast(x=bool_unlabeled, dtype=tf.float32)),
                              y=positive_label)
    #n_unlabeled = tf.subtract(x=tf.cast(x=y_label.shape[0], dtype=tf.float32), y=n_positive)

    ##________________________________________
    ##  LOSS function like "Sigmoid Loss"
    ###____________________
    ###  TODO: I don't know why unbiased_loss's Parameter has minus "self.loss_func(-self.x_in)"
    ### positive_loss = self.loss_func(self.x_in)
    ### unlabeled_loss = self.loss_func(-self.x_in)
    positive_loss = loss_func(logits)
    unlabeled_loss = loss_func(-logits)
    ##________________________________________


    with tf.name_scope(name="Compute_Emperical_Risk"):
        ##------------------------------------------------------------
        ##  Compute Emperical Risk (経験損失)
        ##    - Use Only TensorFrow Operator for Automatic differentiation
        ##    - +,-,*,/ python operator are translated to TensorFlow operator
        ##      - https://stackoverflow.com/a/37901852/9316234
        ###----------------------------------------
        ###  sum(positive_pct * bool_positive / n_positive * positive_loss)
        positive_risk = tf.reduce_sum( input_tensor = positive_pct * tf.cast(x=bool_positive, dtype=tf.float32)
                                                      / n_positive * positive_loss, axis=None)
        ###----------------------------------------
        ###  sum((bool_unlabeled / n_unlabeled - positive_pct * bool_positive / n_positive) * unlabeled_loss)
        negative_risk = tf.reduce_sum(
                input_tensor = ( tf.cast(x=bool_unlabeled, dtype=tf.float32) / n_unlabeled
                                 - positive_pct * tf.cast(x=bool_positive, dtype=tf.float32) / n_positive
                               ) * unlabeled_loss,
                axis=None)
        ###----------------------------------------

        total_risk = positive_risk + negative_risk

    with tf.name_scope(name="nnPU_uPU"):
        if nnPU:
            def manipulate1(total_risk, positive_risk, beta):
                """negative_risk < -beta"""
                total_risk = tf.subtract( x=positive_risk, y=beta )
                x_out = -gamma * negative_risk
                #total_risk = -gamma * negative_risk
                return total_risk, x_out
            def manipulate2(total_risk):
                x_out = total_risk
                return total_risk, x_out
            total_risk, x_out = tf.cond(negative_risk < -beta,
                                        true_fn  = lambda: manipulate1(total_risk, positive_risk, beta),
                                        false_fn = lambda: manipulate2(total_risk) )
        else:
            x_out = total_risk
        loss = total_risk

    return loss

