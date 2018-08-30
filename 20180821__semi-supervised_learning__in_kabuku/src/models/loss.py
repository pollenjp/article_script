

def nnPU_loss(logits, labels, positive_pct = 0.2, beta = 0., gamma = 1., nnPU = True):
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
    #________________________________________________________________________________
    ##      
    loss_func = lambda x: tf.sigmoid(x=x)  # sigmoid: y = 1 / (1 + exp(-x))

    logits  = tf.reshape(tensor=logits, shape=[-1])
    y_label = tf.cast(x=tf.reshape(tensor=labels, shape=[-1]), dtype=tf.float32)
    assert logits.shape == y_label.shape
    
    bool_positive, bool_unlabeled = tf.equal(x=y_label, y=1), tf.equal(x=y_label, y=0)

    ##  count positive and unlabel (Only Support |P| + |U| = |X| or |U| = |X| )
    ##  y_label has 0 (Unlabel) or 1 (Positive)
    n_positive  = tf.cast(x=tf.maximum(x=tf.reduce_sum(input_tensor=y_label), y=1), dtype=tf.float32)
    #  n_unlabeled =  "batch" - n_positive
    n_unlabeled = tf.subtract(x=tf.cast(x=y_label.shape[0], dtype=tf.float32), y=n_positive)

    ##________________________________________
    ##  LOSS function like "Sigmoid Loss"
    ###____________________
    ###  TODO: I don't know why unbiased_loss's Parameter has minus "self.loss_func(-self.x_in)"
    ### positive_loss = self.loss_func(self.x_in)
    ### unlabeled_loss = self.loss_func(-self.x_in)
    positive_loss = loss_func(logits)
    unlabeled_loss = loss_func(-logits)
    ##________________________________________


    ##________________________________________
    ##  Compute Emperical Risk (経験損失)
    ##________________________________________
    ##  Use Only TensorFrow Operator for Automatic differentiation
    ###____________________
    ###  sum(positive_pct * bool_positive / n_positive * positive_loss)
    positive_risk = tf.reduce_sum(
        input_tensor=tf.multiply(
            x=tf.divide(x=tf.multiply(x=positive_pct,
                                      y=tf.cast(x=bool_positive, dtype=tf.float32)),
                        y=n_positive),
            y=positive_loss), axis=None)
    ###____________________
    ###  sum((bool_unlabeled / n_unlabeled - positive_pct * bool_positive / n_positive) * unlabeled_loss)
    negative_risk = tf.reduce_sum(
        input_tensor=tf.multiply(
            x=tf.subtract(x=tf.divide(x=tf.cast(x=bool_unlabeled, dtype=tf.float32),
                                      y=n_unlabeled),
                          y=tf.divide(x=tf.multiply(x=positive_pct,
                                                    y=tf.cast(x=bool_positive, dtype=tf.float32)),
                                      y=tf.cast(n_positive, dtype=tf.float32))),
            y=unlabeled_loss), axis=None)
    #________________________________________

    total_risk = tf.add( x=positive_risk, y=negative_risk )

    if nnPU:
        if negative_risk < -beta:
            print("negative_risk < -beta:")
            total_risk = tf.subtract( x=positive_risk, y=beta )
            loss = -gamma * negative_risk
        else:
            loss = total_risk
    else:
        loss = total_risk

    return loss

