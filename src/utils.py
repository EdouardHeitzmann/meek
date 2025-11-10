import random
import numpy as np

def noise_cvr_array(CVR, candidates, exhaust_sentinel=-1, noise_level=0.05):
    """
    Create a noised version of the CVR array following the same rules as noise_bal.
    
    Args:
        CVR: numpy array where each row is a ballot
        candidates: set of valid candidate IDs
        exhaust_sentinel: value representing exhausted/non-candidate (-1)
        noise_level: fraction of ballots to noise
    
    Returns:
        noised_BAL: numpy array with same shape as CVR but with noised ballots
    """
    noised_BAL = CVR.copy()
    n_ballots = len(CVR)
    n_to_noise = int(noise_level * n_ballots)
    
    # Randomly select indices to noise
    noised_indices = random.sample(range(n_ballots), k=n_to_noise)
    
    for idx in noised_indices:
        ballot = CVR[idx].copy()
        noised_BAL[idx] = noise_single_cvr(ballot, candidates, exhaust_sentinel)
    
    return noised_BAL

def noise_single_cvr(ballot, candidates, exhaust_sentinel=-1):
    """
    Apply noise to a single CVR ballot following the original noise_bal logic.
    """
    # Convert ballot to list of non-exhaust candidates (equivalent to the old tuple format)
    active_positions = ballot != exhaust_sentinel
    if not np.any(active_positions):
        # All exhausted ballot - treat as empty
        bal_list = []
    else:
        bal_list = ballot[active_positions].tolist()
    
    # Check if this is an "exhausted" ballot (equivalent to old (6,) case)
    if len(bal_list) == 0 or (len(bal_list) == 1 and bal_list[0] not in candidates):
        return random.choice([
            np.array([1, 2, exhaust_sentinel], dtype=ballot.dtype),
            np.array([2, exhaust_sentinel, exhaust_sentinel], dtype=ballot.dtype),
            np.array([3, exhaust_sentinel, exhaust_sentinel], dtype=ballot.dtype),
            np.array([4, 1, exhaust_sentinel], dtype=ballot.dtype),
            np.array([5, 1, exhaust_sentinel], dtype=ballot.dtype)
        ])
    
    noise_type = random.choice([1, 2, 3, 4, 5])
    
    if noise_type == 2:  # delete ranking
        if len(bal_list) == 1:
            noise_type = 1
        else:
            del_pos = random.randint(0, len(bal_list) - 1)
            del bal_list[del_pos]
    
    if noise_type == 1:  # insert ranking
        available_cands = [c for c in candidates if c not in bal_list]
        if available_cands:  # only insert if there are candidates available
            cand_to_insert = random.choice(available_cands)
            insert_pos = random.randint(0, len(bal_list))
            bal_list.insert(insert_pos, cand_to_insert)
    
    if noise_type == 3:  # permute two rankings
        if len(bal_list) == 1:
            noise_type = 4
        else:
            indices = random.sample(range(len(bal_list)), 2)
            bal_list[indices[0]], bal_list[indices[1]] = bal_list[indices[1]], bal_list[indices[0]]
    
    if noise_type == 4:  # switch one ranking with another non-ranked candidate
        available_cands = [c for c in candidates if c not in bal_list]
        if available_cands and len(bal_list) > 0:
            cand_to_switch_in = random.choice(available_cands)
            switch_pos = random.randint(0, len(bal_list) - 1)
            bal_list[switch_pos] = cand_to_switch_in
    
    if noise_type == 5:  # become "exhausted" (equivalent to old (6,))
        # Create a ballot with one invalid candidate
        result = np.full(ballot.shape, exhaust_sentinel, dtype=ballot.dtype)
        result[0] = max(candidates) + 1  # Use a candidate ID not in the race
        return result
    
    # Convert back to array format
    result = np.full(ballot.shape, exhaust_sentinel, dtype=ballot.dtype)
    for i, cand in enumerate(bal_list[:len(ballot)]):
        result[i] = cand
    
    return result
