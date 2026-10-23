import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder

def preprocess_dutch_data(seed):
    
    np.random.seed(seed)
    data_path = '/data/share/dutch/'
    raw_data = pd.read_csv(data_path + 'dutch.csv')
    
    categorical_features = ['age', 'household_position', 'household_size', 'citizenship', 'country_birth', 'edu_level',
                            'economic_status', 'cur_eco_activity', 'marital_status']
    new_categorical_data = {}
    for original_categorical_feature in categorical_features:
        oh_enc = OneHotEncoder(sparse_output=False)
        new_categorical_onehot_features = oh_enc.fit_transform(raw_data[[original_categorical_feature]])
        for i, value in enumerate(set(raw_data[original_categorical_feature])):
            new_categorical_data[f'{original_categorical_feature}={value}'] = new_categorical_onehot_features[:, i]
    new_categorical_data = pd.DataFrame(new_categorical_data)
    
    raw_data = pd.concat([raw_data, new_categorical_data], axis=1)
    for categorical_feature in categorical_features:
        del raw_data[categorical_feature]
    
    feature_data = deepcopy(raw_data)
    del feature_data['sex']
    del feature_data['occupation']
    feature_data['prev_residence_place'] -= 1
    
    all_features = np.array(feature_data).astype(float)
    all_sensitives = np.array(raw_data['sex'] == 'male').astype(float).flatten()
    all_labels = np.array(raw_data['occupation']).astype(float).flatten()
                    
    random_ids = np.random.permutation(len(raw_data))
    train_ids = random_ids[:int(0.8*len(raw_data))]
    test_ids = random_ids[int(0.8*len(raw_data)):]

    train_features, train_sensitives, train_labels = all_features[train_ids], all_sensitives[train_ids], all_labels[train_ids]
    test_features, test_sensitives, test_labels = all_features[test_ids], all_sensitives[test_ids], all_labels[test_ids]

    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    return train_features, test_features, train_labels, test_labels, train_sensitives, test_sensitives
