import tensorflow as tf
from tqdm import tqdm


@tf.function
def _compute_apply_gradients(model, x, optimizer, loss_fcn):
    with tf.GradientTape() as tape:
        loss = loss_fcn(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_model(model, loss_fcn, train_dataset, eval_dataset, optimizer, iterations, batch_size, processed=True,
                iter_check=1000):
    train_dataset = train_dataset.batch(batch_size).repeat()
    eval_dataset = eval_dataset.batch(batch_size).repeat(1)

    for iter, train_batch in enumerate(train_dataset):
        if iter > iterations:
            break

        train_loss = _compute_apply_gradients(model, train_batch, optimizer, loss_fcn)

        if iter % iter_check == 0:
            loss_mean = tf.keras.metrics.Mean()
            for eval_batch in eval_dataset:
                loss_mean(loss_fcn(model, eval_batch))
            print("Iter: {}, Train loss: {}, Eval loss: {}".format(iter, train_loss, loss_mean.result()))





        # if epoch % 1 == 0:
        #     loss = tf.keras.metrics.Mean()
        #     for test_x in eval_dataset:
        #         loss(loss_fcn(model, test_x))
        #     elbo = -loss.result()
        #
        #     print('Epoch: {}, Test set ELBO: {}'.format(epoch, elbo))


