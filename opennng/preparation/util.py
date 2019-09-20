import sys
import requests
import os


def download_dataset(url, filename, name):
    print("Downloading dataset `{}`.".format(name))

    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}{}] {}%'.format('=' * (done-1), ">",  '.' * (50 - done), done*2))
                sys.stdout.flush()
    sys.stdout.write('\n\n')


def make_dirs(dataset, from_noise=True):
    temp_path = os.path.join("data", "temp")
    data_path = os.path.join("data", dataset, "raw")
    temp_data_path = os.path.join(temp_path, dataset + ".tar.gz")

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    if not os.path.exists(data_path):
        if from_noise:
            os.makedirs(os.path.join(data_path, "train"))
            os.makedirs(os.path.join(data_path, "valid"))
        else:
            os.makedirs(os.path.join(data_path, "train_X"))
            os.makedirs(os.path.join(data_path, "valid_X"))
            os.makedirs(os.path.join(data_path, "train_y"))
            os.makedirs(os.path.join(data_path, "valid_y"))

    return temp_path, data_path, temp_data_path