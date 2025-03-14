import pandas as pd

def load_and_merge_datasets(train_path, valid_path, test_path):

    train_df = pd.read_csv(train_path)[:256]
    valid_df = pd.read_csv(valid_path)[:32]
    test_df = pd.read_csv(test_path)[:32]
    
    train_df['train'], train_df['valid'], train_df['test'] = 1, 0, 0
    valid_df['train'], valid_df['valid'], valid_df['test'] = 0, 1, 0
    test_df['train'], test_df['valid'], test_df['test'] = 0, 0, 1
    
    merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    return merged_df

merged_df = load_and_merge_datasets(
    'internal_dataset/trans_new_train.csv',
    'internal_dataset/trans_new_val.csv',
    'internal_dataset/trans_new_test.csv'
)

merged_df.to_csv('internal_dataset/trans_new_small.csv', index=False)
