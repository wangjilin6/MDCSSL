import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def get_train_test():
    data_label = np.load('/data/WangJilin/data/MIT-BIH/data_used/data_label_5class.npy')
    data = data_label[:,0].reshape(-1,1)
    labels = data_label[:,1].reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=3407, shuffle=True, stratify=labels)
    # print(x_test[:10])

    rus = RandomUnderSampler(sampling_strategy={0:8000},random_state=3407)
    x_train_balance, y_train_balance = rus.fit_resample(x_train,y_train)
    ros = RandomOverSampler(sampling_strategy={3: 1500}, random_state=3407)
    x_train_balance, y_train_balance = ros.fit_resample(x_train_balance, y_train_balance)

    return x_train_balance.reshape(-1), x_test.reshape(-1), y_train_balance, y_test

if __name__ == '__main__':
    x,y,z,c = get_train_test()
    print(x.shape,y.shape,z.shape,c.shape)