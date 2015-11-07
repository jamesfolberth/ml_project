"""
Just some utility functions
"""

def split_list(l, splits):
    """
    Split a list according to the iterable splits, giving the split percentages
    """
    lists = []
    splits = map(int, splits)
    if sum(splits) != 100:
        raise ValueError("splits should sum to 100")

    if splits[0] == 100:
        splits = [100, 0]

    ind_start = 0
    for split in splits:
        ind_end = ind_start + int(float(split)/100.*float(len(l)))

        if ind_end == len(l) - 1: # clean up last index
            ind_end = len(l)

        lists.append(l[ind_start:ind_end])
        ind_start = ind_end

    return lists

