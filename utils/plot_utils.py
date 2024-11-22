import os
import pyjson5

import pickle as pkl

import pandas as pd

import torch
import numpy as np

from scipy.special import betainc

import mpmath as mp
mp.mp.dps = 100


# GS
# 256 message size in bits, 128 repetitions
#Threshold for fpri 1e-06     :0.60546875  
#Threshold for fpri 1e-16     :0.6796875   
#Threshold for fpri 1e-32     :0.755859375 
#Threshold for fpri 1e-64     :0.85546875
GS_THRESHOLDS_LARGE = {
    "1e-6": 0.60546875,
    "1e-16": 0.6796875,
    "1e-32": 0.755859375,
    "1e-64": 0.85546875
}

# 128 message size in bits, 128 repetitions
#Threshold for fpri 1e-06     :0.6484375   
#Threshold for fpri 1e-16     :0.75        
#Threshold for fpri 1e-32     :0.8515625   
#Threshold for fpri 1e-64     :0.97265625

#GS_THRESHOLDS_SMALL = {
#    "1e-6": 0.6484375,
#    "1e-16": 0.75,
#    "1e-32": 0.8515625,
#    "1e-64": 0.97265625
#}

GS_THRESHOLDS_SMALL = {
    "1e-6": 0.71484375,
    "1e-16": 0.79296875,
    "1e-32": 0.8828125,
    "1e-64": 0.984375
}

TR_THRESHOLDS_SD21_Anime = {
    "1e-1": 0.14274865156269712,
    "1e-2": 0.024656736901611347,
    "1e-3": 0.005552058156919453
}

TR_THRESHOLDS_SDXL = {
    #"1e-1": 0.05565081310329875,
    "1e-1": 0.1349257019964569,
    #"1e-2": 0.025855301891134466,
    "1e-2": 0.0261008646426459,
    #"1e-3": 0.00533133476249702
    "1e-3": 0.005532768367901984
}

TR_THRESHOLDS_Pixart = {
    "1e-1": 0.15808728685433943,
    "1e-2": 0.015917123872865434,
    "1e-3": 0.00048163976257239907
}

TR_THRESHOLDS_Flux = {
    "1e-1": 0.14287198350046235,
    "1e-2": 0.02185009852518841,
    "1e-3": 0.0007434650977099187
}

TR_THRESHOLD_FOR_MODEL = {
    "stabilityai/stable-diffusion-2-1-base": TR_THRESHOLDS_SD21_Anime,
    "stabilityai/stable-diffusion-xl-base-1.0": TR_THRESHOLDS_SDXL,
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": TR_THRESHOLDS_Pixart,
    "black-forest-labs/FLUX.1-dev": TR_THRESHOLDS_Flux
}




# -------------------------------------------------------------------- GS IDENTIFICATION --------------------------------------------------------------------

def eval_identification(frame_, amount_users, eager_push_to_gpu=True):

    # unpack the message_bits_str_recovered
    frame_["message_bits_str_recovered"] = frame_.apply(lambda row: row["message_bits_str_recovered"][0] if row["message_bits_str_recovered"] is not None else None, axis=1)

    recovered_messages = frame_["message_bits_str_recovered"].tolist()
    original_messages_unique = frame_["message_bits_str_initial"].unique().tolist()

    # --------------------------------- CREATE random user Message IDs ---------------------------------
    print("\t\tCREATE random user Message IDs")
    all_messages = original_messages_unique.copy()
    
    def generate_unique_strings(given_messages, length):
        """take the unique retrieved messages and generate random bits for the remaining users"""
        given_messages = np.array([[int(char) for char in string] for string in given_messages], dtype=np.uint8)
        random_bits = np.random.randint(2, size=(amount_users - len(given_messages), length), dtype=np.uint8)
        return np.concatenate((given_messages, random_bits), axis=0)

    # Generate unique strings
    message_length = len(all_messages[0])  # 256
    amount_of_unique_retrieved_messages = len(all_messages)  # amount of target images, so 100 for 100-10-10
    all_messages_array = torch.tensor(generate_unique_strings(all_messages,
                                                              message_length))

    # --------------------------------- CREATE MATRIX: Compare all recovered messages (axis 0) with all original messages (axis 1) ---------------------------------
    print("\t\tCREATE MATRIX: Compare all recovered messages with all original messages")
    
    # Convert binary strings to PyTorch tensors of integers
    recovered_array = torch.tensor([[int(bit) for bit in msg] for msg in recovered_messages], dtype=torch.uint8)  # shape = (10k, 256) for enhanced attack (for 100-10-10)
    
    # Move tensors to GPU
    recovered_array = recovered_array.cuda() if eager_push_to_gpu else recovered_array
    all_messages_array = all_messages_array.cuda() if eager_push_to_gpu else all_messages_array

    # Initialize accuracy matrix on GPU
    accuracy_matrix = torch.empty((recovered_array.shape[0], all_messages_array.shape[0]),
                                  device='cuda' if eager_push_to_gpu else "cpu")  # shape
    accuracy_matrix = accuracy_matrix.to(torch.uint8)
    
    # Compute bitwise XOR and count matching bits on GPU
    #rows = []
    for i in range(recovered_array.shape[0]):
        if not eager_push_to_gpu:
            xor_result = torch.bitwise_xor(recovered_array[i].unsqueeze(0).to("cuda"), all_messages_array.to("cuda"))
        else:
            xor_result = torch.bitwise_xor(recovered_array[i].unsqueeze(0), all_messages_array)
        #rows.append(torch.argmin(xor_result, dim=1))
        accuracy_matrix[i, :] = torch.sum(xor_result, dim=1).cpu() if not eager_push_to_gpu else torch.sum(xor_result, dim=1)

        if not eager_push_to_gpu and i % 100 == 0:
            print(f"\t\t\t{i}/{recovered_array.shape[0]}")


    # Tensor of shape len(recovered_messages)
    matching_index_for_recovered_message = torch.argmin(accuracy_matrix, dim=1)
    #matching_index_for_recovered_message = torch.stack(rows, dim=0)

     # add column to frame
    frame_[f"{amount_users}_matching_index_for_recovered_message"] = matching_index_for_recovered_message.cpu().tolist()
    frame_[f"{amount_users}_matching_message_for_recovered_message"] = [''.join([str(m) for m in all_messages_array[i].cpu().tolist()]) for i in matching_index_for_recovered_message.cpu().tolist()]
    frame_[f"{amount_users}_successful_match"] = frame_["message_bits_str_initial"] == frame_[f"{amount_users}_matching_message_for_recovered_message"]
    frame_[f"{amount_users}_successful_match_int"] = frame_[f"{amount_users}_successful_match"].astype(int)

    del all_messages_array
    del recovered_array

    return frame_


#FPR_FIX = 1e-32
INTERESTING_FPRs_GS = [10**-i for i in [
    #3, 6, 9, 12,
    6,
    16, #24,
    32,
    64
    ]]

def get_GS_thresholds(num_bits=256, NUM_USERS=100*1000):
    # Number of bits in the WM message
    #K = 256
    K = num_bits
    N = NUM_USERS
    
    def beta_func(τ, k=K):
        a = τ + 1
        b = k - τ
        return betainc(a, b, 0.5)

    # Generating thresholds and their corresponding single-user FPR
    thresholds = range(K//2, K + 1)
    single_user_FPRs = [beta_func(τ) for τ in thresholds]

    # Calculating multi-user FPR
    single_user_FPRs = np.array(single_user_FPRs)  # Ensure it's a numpy array if not already

    # Convert single_user_FPRs to mpmath floats for higher precision
    single_user_FPRs_mp = [mp.mpf(fpr) for fpr in single_user_FPRs]

    # Compute the result with high precision
    multi_user_FPRs = [1 - mp.exp(-N * fpr) for fpr in single_user_FPRs_mp]
    
    
    def find_first_index_below_threshold(values, a):
        return next((i for i, x in enumerate(values) if x < a), None)

    # SINGLE
    FPRs_GS = [beta_func(τ) for τ in thresholds]
    THRESHOLD_INDEXES_FOR_FPRs_GS = {fpri: find_first_index_below_threshold(FPRs_GS, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS.items()}
    
    print("single:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")
        
    # MULTI
    FPRs_GS_MULTI = multi_user_FPRs
    THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI = {fpri: find_first_index_below_threshold(FPRs_GS_MULTI, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS_MULTI = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS_MULTI.items()}
    print("multi:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")

    return {"INTERESTING_FPRs_GS": INTERESTING_FPRs_GS,
            "THRESHOLD_INDEXES_FOR_FPRs_GS": THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FOR_FPRs_GS": THRESHOLD_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FLOAT_FOR_FPRs_GS": THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI}
    
    
def filter_identification_on_threshold(frame___, amount_users: int, THRESHOLD_FLOAT_FOR_FPRs_GS: dict):
    amount_users = int(amount_users)
    for fpr in INTERESTING_FPRs_GS:
        frame___[f"{fpr}_{amount_users}_successful_match"] = frame___.apply(lambda row: row[f"{amount_users}_successful_match"] if row["bit_accuracy"] > THRESHOLD_FLOAT_FOR_FPRs_GS[fpr] else False, axis=1)
        frame___[f"{fpr}_{amount_users}_successful_match_int"] = frame___[f"{fpr}_{amount_users}_successful_match"].astype(int)
    return frame___


def expand_df_with_thresholds(df, suffix="successful_match_int"):
    """
    Expands a DataFrame by transforming columns with a specific suffix
    (e.g., 'successful_match_int') into a new column with the first threshold as a value.
    
    Args:
        df (pd.DataFrame): The original DataFrame to transform.
        suffix (str): The column suffix to search for and transform (default is 'successful_match_int').

    Returns:
        pd.DataFrame: The transformed DataFrame with expanded rows.
    """
    # Extract all columns that match the pattern
    successful_match_int_cols = [col for col in df.columns if col.endswith(suffix)]
    
    # Create a new column for the first threshold (e.g., 1e-16, 1e-24, etc.) and copy the rows for each one
    dfs = []
    for col in successful_match_int_cols:
        threshold = col.split('_')[0]  # Extract the first threshold (e.g., 1e-16, 1e-24)
        
        # Create a new DataFrame with this threshold value and the corresponding "successful_match_int" values
        df_copy = df.copy()
        df_copy['threshold'] = threshold  # Create a new column for the first threshold
        df_copy['successful_match_int'] = df[col]  # Map the "successful_match_int" value from the respective column
        
        # Append this DataFrame to the list of DataFrames
        dfs.append(df_copy)
    
    # Concatenate all DataFrames into one
    df_expanded = pd.concat(dfs, ignore_index=True)
    
    # Drop the old "successful_match_int" columns
    df_expanded = df_expanded.drop(columns=successful_match_int_cols)
    
    return df_expanded
