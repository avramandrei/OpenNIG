import os
import numpy as np
import cv2


def generate_samples(model, num_sample, samples_save_path):
    print("\nGenerating: {} samples with model {}...".format(num_sample, model.name))

    for i in range(num_sample):
        sample = np.squeeze(model.generate()) * 255

        cv2.imwrite(os.path.join(samples_save_path, "sample_{}.jpg".format(i+1)), sample)

        print("\tSample {}: generated".format(i+1))

    print("Finished!")
    