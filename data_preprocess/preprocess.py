import pandas as pd

def restructure_dataset(df):

    grouped = [df.iloc[i:i+3] for i in range(0, len(df), 3)]
    
    new_data = []
    new_id = 1
    for group in grouped:
        if len(group) < 3:
            continue

        id1, rxn1 = group.iloc[0]['id'], group.iloc[0]['rxn_smiles']
        id2, rxn2 = group.iloc[1]['id'], group.iloc[1]['rxn_smiles']
        id3, rxn3 = group.iloc[2]['id'], group.iloc[2]['rxn_smiles']
        
        A, B = rxn1.split('>>')
        _, C = rxn2.split('>>')
        _, D = rxn3.split('>>')
        
        new_data.append({
            'original_id': f"{id1}_{id2}_{id3}",  
            'new_id': new_id,  
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'A_to_B': f"{A}>>{B}",
            'A_to_B_to_C': f"{A}>>{B}>>{C}",
            'A_to_B_to_C_to_D': f"{A}>>{B}>>{C}>>{D}"
        })
        new_id += 1  
    
    return pd.DataFrame(new_data)

df = pd.read_csv('datasets/Trans_G2S_train.csv')  
new_df = restructure_dataset(df)
new_df.to_csv('datasets/trans_new_train.csv', index=False)  
