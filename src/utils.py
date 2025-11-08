import random

def project_FPV(bal, hopeful, g_cand = 6):
    list_bal = list(bal)
    if len(list_bal) == 0:
        return g_cand # Jill Stein
    while bal[0] not in hopeful:
        bal = bal[1:]
        if len(bal) == 0:
            return g_cand # Jill Stein
    return bal[0]

def noise_bal(bal, cands = {1,2,3,4,5}):
    if bal == (6,):
        return random.choice([(1,2), (2,), (3,), (4,1), (5,1)])
    else:
        noise_type = random.choice([1,2,3,4,5]) 
        if noise_type == 2: # delete ranking
            if len(bal) == 1:
                noise_type = 1
            else:
                del_pos = random.randint(0, len(bal)-1)
                new_bal = list(bal)
                del new_bal[del_pos]
                return tuple(new_bal)
        if noise_type == 1:# insert ranking
            # our ballots are all short, so no need to worry about having no cands to insert
            cand_to_insert = random.choice([c for c in cands if c not in bal])
            new_bal = list(bal)
            insert_pos = random.randint(0, len(new_bal))
            new_bal.insert(insert_pos, cand_to_insert)
            return tuple(new_bal)
        if noise_type == 3: # permute two rankings
            if len(bal) == 1:
                noise_type = 4
            else:
                bal_list = list(bal)
                indices = random.sample(range(len(bal)), 2)
                bal_list[indices[0]], bal_list[indices[1]] = bal_list[indices[1]], bal_list[indices[0]]
                return tuple(bal_list)
            return tuple(sorted(set(bal).intersection(cands)))
        if noise_type == 4: #switch one ranking with another non-ranked candidate
            cand_to_switch_in = random.choice([c for c in cands if c not in bal])
            switch_pos = random.randint(0, len(bal)-1)
            bal_list = list(bal)
            bal_list[switch_pos] = cand_to_switch_in
            return tuple(bal_list)
        if noise_type == 5: # become Jill Stein
            return (6,)
    raise Exception("Should not reach here")