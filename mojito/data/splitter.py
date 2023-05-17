def split_data(sorted_user_interactions):
    """
    Split data into train/valid/test
    The last item is for test
    The second last item is for validation
    The remaining items are for train
    :param sorted_user_interactions: dictionary, values are
           tuples (iid, timestamp)
    :return:
    """
    train_set = {}
    valid_set = {}
    test_set = {}
    for uid, interactions in sorted_user_interactions.items():
        nfeedback = len(interactions)
        if nfeedback < 3:
            train_set[uid] = interactions
            valid_set[uid] = []
            test_set[uid] = []
        else:
            train_set[uid] = interactions[:-2]
            valid_set[uid] = [interactions[-2]]
            test_set[uid] = [interactions[-1]]
    return train_set, valid_set, test_set
