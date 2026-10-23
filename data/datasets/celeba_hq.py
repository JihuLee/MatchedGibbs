import numpy as np
import pandas as pd

def preprocess_celeba_data(seed, target_id=2, sensitive_id=20): 
    np.random.seed(seed)
    data_path = '/data/share/celeba_mask_hq/CelebAMask-HQ/'

    df = pd.read_csv(data_path + 'CelebAMask-HQ-attribute-anno.csv')
    
    label_np = df.drop(columns='filename').replace(-1, 0).values

    n = label_np.shape[0]
    random_ids = np.random.permutation(n)
    train_ids = random_ids[:int(0.8*n)]
    test_ids = random_ids[int(0.8*n):]
    
    train_labels = label_np[train_ids, target_id].astype(int)
    test_labels = label_np[test_ids, target_id].astype(int)

    if sensitive_id == 20:
        sensitives = -label_np[:, sensitive_id].astype(int) + 1
    else:
        sensitives = label_np[:, sensitive_id].astype(int)
    train_sensitives = sensitives[train_ids]
    test_sensitives = sensitives[test_ids]

    features = np.load(data_path + 'embeddings.npz')['embeddings']

    train_features = features[train_ids]
    test_features = features[test_ids]

    train_labels = np.expand_dims(train_labels, axis=1)
    test_labels = np.expand_dims(test_labels, axis=1)

    return train_features, test_features, train_labels, test_labels, train_sensitives, test_sensitives