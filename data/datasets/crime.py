import numpy as np
import pandas as pd

def preprocess_crime_data(fold_id, binarize=True, balancing_train_data=False):
    drop_sensitive = True
    
    def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1, drop_sensitive=False):
        data_dir = '/data/share/crime/'

        names = []
        with open(data_dir + 'communities.names', 'r') as file:
            for line in file:
                if line.startswith('@attribute'):
                    names.append(line.split(' ')[1])

        data = pd.read_csv(data_dir + 'communities.data', names=names, na_values=['?'])

        to_drop = ['state', 'county', 'community', 'fold', 'communityname']
        data.fillna(0, inplace=True)
        data = data.sample(frac=1, replace=False).reset_index(drop=True)

        folds = data['fold'].astype(int)

        y = data[label].values
        to_drop += [label]

        z = data[sensitive_attribute].values
        if drop_sensitive:
            to_drop += [sensitive_attribute]

        data.drop(to_drop + [label], axis=1, inplace=True)

        for n in data.columns:
            data[n] = (data[n] - data[n].mean()) / data[n].std()

        x = np.array(data.values)
        
        train_features, train_labels, train_sensitives = x[folds != fold], y[folds != fold], z[folds != fold]
        test_features, test_labels, test_sensitives = x[folds == fold], y[folds == fold], z[folds == fold]

        return train_features, test_features, train_labels, test_labels, train_sensitives, test_sensitives
    
    x_train, x_test, y_train, y_test, s_train, s_test = read_crimes(fold=fold_id, drop_sensitive=drop_sensitive)

    if binarize:
        y_train_median = np.median(y_train)
        y_train = (y_train >= y_train_median).astype(float).flatten()
        y_test = (y_test >= y_train_median).astype(float).flatten()

    s_train_median = np.median(s_train)
    s_train = (s_train >= s_train_median).astype(float).flatten()
    s_test = (s_test >= s_train_median).astype(float).flatten()

    if balancing_train_data:
        groups = {}
        for i, (y, s) in enumerate(zip(y_train, s_train)):
            key = (int(y), int(s))
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        min_count = min(len(indices) for indices in groups.values())
        
        selected_indices = []
        for indices in groups.values():
            selected_indices.extend(np.random.choice(indices, min_count, replace=False))
        
        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]
        s_train = s_train[selected_indices]
        
        shuffle_idx = np.random.permutation(len(x_train))
        x_train = x_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        s_train = s_train[shuffle_idx]
    s_train = (s_train >= s_train_median).astype(float)
    s_test = (s_test >= s_train_median).astype(float)
    

    y_train = np.expand_dims(y_train, 1)
    y_test = np.expand_dims(y_test, 1)
    
    return x_train, x_test, y_train, y_test, s_train, s_test