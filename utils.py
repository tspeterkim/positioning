import numpy as np

def find_sampling_freq(path, pos):
    with open(path+pos+'/1_android.sensor.accelerometer.data.csv') as f:
        freq = 0
        lower = float(f.readline().split(',')[0])
        for i, line in enumerate(f):
            upper = float(line.split(',')[0])
            second = (upper-lower)/1000
            if 0.99 <= second and second <= 1.01:
                freq = i+1
                break
        return freq


def create_dataset(path, timesteps):
    X = []
    y = []
    positions = ['layingdown','sitting','standing','walking']
    print(path)
    for ip, p in enumerate(positions):
        data = []
        with open(path+p+'/1_android.sensor.accelerometer.data.csv') as f:
            # ts = find_sampling_freq(path, p)
            # if ts == 0:
            ts = timesteps
            # print('\tsampling freq = %i'% ts)
            row = []
            for i, line in enumerate(f):
                row.append([float(x) for x in line.split(',')[1:4]])
                if len(row) == ts:
                    data.append(np.stack(row))
                    row = []
        # Zero pad to account for different sampling frequency
        # max_len = np.amax([x.shape[0] for x in data])
        # print('\tmax sampling frequency = %i' % max_len)
        # data = [np.lib.pad(x, ((0,0),(0,max_len-x.shape[1])), 'constant') for x in data]
        data = np.stack(data)
        print("\t%s: %i examples loaded" % (p, data.shape[0]))
        X.append(data)
        y.append(np.zeros(data.shape[0])+ip)

    X = np.vstack(X)
    y = np.concatenate(y)
    # print(X.shape, y.shape)
    return (X, y)


def dataloader(train_dataset, batch_size):
    X, y = train_dataset
    arr = np.arange(X.shape[0])
    np.random.shuffle(arr)
    batches, batch_x, batch_y = [], [], []
    for a, i in enumerate(arr):
        batch_x.append(X[i])
        batch_y.append(y[i])
        if a == len(arr)-1 or len(batch_x) == batch_size:
            batches.append((np.stack(batch_x).astype(np.float32), np.array(batch_y).astype(int)))
            batch_x, batch_y = [], []
    return batches


if __name__ == '__main__':
    print("sampling frequency = %i" % find_sampling_freq('data/test/','standing'))
