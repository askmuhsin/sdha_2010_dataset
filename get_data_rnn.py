import numpy as np
from keras.utils import to_categorical

dir_seq = './data/sequence_gen/rnn_sequence_1.npy'
seq = np.load(dir_seq)
seq = seq.item()
max_length = 120*100  # len of encoded data per seq * max frames per Sequence

def explore():
    sequence_length = []
    number_of_sequence = 0
    for key, _ in seq.items():
        for key_in, val_in in seq[key].items():
            sequence_length.append(val_in.shape[0])
            number_of_sequence+=1
    print("Total number of sequences : ", number_of_sequence)
    print("max frames = {}, min frames = {}".format(max(sequence_length),min(sequence_length)))

def stackAndPad(array):
    temp = np.concatenate(array, axis=0)
    pad_array = np.zeros((max_length - temp.shape[0]))
    padded = np.hstack((pad_array, temp))
    assert padded.shape[0]==max_length, "Sequence overflow"
    return padded

def load_data_sequencial():
    X, y = [], []
    for key, _ in seq.items():
        for key_in, val_in in seq[key].items():
            X.append(stackAndPad(val_in))
            y.append(int(key.split('_')[1]))

    X, y = np.asarray(X), np.asarray(y)
    y_one_hot = to_categorical(y, num_classes=6)
    return X, y

def main():
    explore()

if __name__ == '__main__':
    main()
