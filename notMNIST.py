# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

import os
import pickle
import sys
import tarfile
from urllib import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score

from Util.StringUtil import printProgress

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = 'Data/'  # Change me to store data elsewhere


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # Normalization ongoing
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def measure_overlap(dataset1, dataset2, exact=True):
    overlap_counter = 0.
    j = 0
    for data1 in dataset1:
        for data2 in dataset2:
            if exact:
                if np.array_equiv(data1, data2):
                    overlap_counter += 1
                    break
            else:
                if np.allclose(data1, data2):
                    overlap_counter += 1
                    break
        j += 1
        printProgress(j, len(dataset1), prefix='Progress:', suffix='Complete', barLength=50)
    print "Overlap percentage:", len(dataset1) / overlap_counter


def create_sanitized_dataset(dataset1, label1, dataset2, img_size, exact=True):
    overlap_counter = 0.
    j = 0
    sanitized_dataset, sanitized_label = make_arrays(len(dataset1), img_size)
    for i in xrange(0, len(dataset1)):
        control = False
        for data2 in dataset2:
            if exact:
                if np.array_equiv(dataset1[i], data2):
                    overlap_counter += 1
                    control = True
                    break
            else:
                if np.allclose(dataset1[i], data2):
                    overlap_counter += 1
                    break
        if not control:
            np.append(sanitized_dataset, dataset1[i])
            np.append(sanitized_label, label1[i])
        j += 1
        printProgress(j, len(dataset1), prefix='Progress:', suffix='Complete', barLength=50)
    print "Overlap percentage:", len(dataset1) / overlap_counter
    print "New size:", len(sanitized_dataset)
    return sanitized_dataset, sanitized_label


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

n_train_samples = 10000


def train_model(train_dataset, train_labels, n_train_samples):
    clf = linear_model.LogisticRegression()
    nsamples, nx, ny = train_dataset.shape
    d2_train_dataset = train_dataset.reshape((nsamples, nx * ny))
    clf.fit(d2_train_dataset[:n_train_samples], train_labels[:n_train_samples])
    return clf


def test_model(test_dataset, test_labels, clf):
    nsamples, nx, ny = test_dataset.shape
    d2_test_dataset = test_dataset.reshape((nsamples, nx * ny))
    predicted_labels = clf.predict(d2_test_dataset)
    print accuracy_score(test_labels, predicted_labels)


clf = train_model(train_dataset, train_labels, n_train_samples)
test_model(test_dataset, test_labels, clf)
# measure_overlap(valid_dataset, test_dataset, exact=True)
# measure_overlap(valid_dataset, test_dataset, exact=False)
sanitized_valid_dataset, sanitized_valid_labels = create_sanitized_dataset(valid_dataset, valid_labels, train_dataset,
                                                                           image_size)
sanitized_test_dataset, sanitized_test_labels = create_sanitized_dataset(test_dataset, test_labels, train_dataset,
                                                                         image_size)
plt.imshow(test_dataset[0])
plt.show()

pickle_file = os.path.join(data_root, 'notMNIST.pickle')
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        'sanitazed_valid_dataset': sanitized_valid_dataset,
        'sanitazed_valid_labels': sanitized_valid_labels,
        'sanitized_test_dataset': sanitized_test_dataset,
        'sanitazed_test_labels': sanitized_test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
