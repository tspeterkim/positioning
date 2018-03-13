import numpy as np

def create_dataset(path, timesteps):
    X = []
    y = []
    positions = ['layingdown','sitting','standing','walking']
    for ip, p in enumerate(positions):
        data = []
        with open(path+p+'/1_android.sensor.accelerometer.data.csv') as f:
            row = []
            for i, line in enumerate(f):
                row.append([float(x) for x in line.split(',')[1:4]])
                if len(row) == timesteps:
                    data.append(np.stack(row))
                    row = []
        data = np.stack(data)
        print("%s : %i examples loaded" % (p, data.shape[0]))
        X.append(data)
        y.append(np.zeros(data.shape[0])+ip)

    X = np.vstack(X)
    y = np.concatenate(y)
    # print(X.shape, y.shape)
    return (X, y)

def dataloader(train_dataset, batch_size):
    # for i, (images, labels) in enumerate(train_loader)
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
