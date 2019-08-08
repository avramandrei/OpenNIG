from src.models.cvae import CVAESmall
import argparse
from src.preparation.prepare import load_data
import tensorflow as tf
from src.util.losses import compute_vae_loss
from src.util.trainer import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--eval_data", type=str)
    parser.add_argument("--processed", type=bool, default=False)

    args = parser.parse_args()

    config = {
        "latent_dim": 50
    }

    if args.model == "CVAE":
        model = CVAESmall((28, 28, 1))
        loss_fcn = compute_vae_loss
    else:
        raise ValueError("Selected model not found")

    if args.train_data is not None:
        train_dataset = load_data(args.train_data)
    else:
        raise ValueError("Selected train data not found")

    if args.eval_data is not None:
        eval_dataset = load_data(args.eval_data)
    else:
        raise ValueError("Selected train data not found")

    processed = args.processed

    optimizer = tf.keras.optimizers.Adam(1e-4)
    iterations = 10000

    train_model(model, loss_fcn, train_dataset, eval_dataset, optimizer, iterations, 32)

