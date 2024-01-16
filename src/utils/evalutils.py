import segeval as seg

def get_seg_boundaries(classifications, sentences_length = None):
    """
    :param list of tuples, each tuple is a sentence and its class (1 if it the sentence starts a segment, 0 otherwise).
    e.g: [(this is, 0), (a segment, 1) , (and another one, 1)
    :return: boundaries of segmentation to use for pk method. For given example the function will return (4, 3)
    """
    curr_seg_length = 0
    boundaries = []
    for i, classification in enumerate(classifications):
        is_split_point = bool(classifications[i])
        add_to_current_segment = 1 if sentences_length is None else sentences_length[i]
        curr_seg_length += add_to_current_segment
        if (is_split_point):
            boundaries.append(curr_seg_length)
            curr_seg_length = 0

    return boundaries

def pk(h, gold, window_size=-1):
    """
    :param gold: gold segmentation (item in the list contains the number of words in segment) 
    :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
    :param window_size: optional 
    :return: accuracy
    """
    if window_size != -1:
        false_seg_count, total_count = seg.pk(h, gold, window_size=window_size, return_parts=True)
    else:
        false_seg_count, total_count = seg.pk(h, gold, return_parts=True)

    if total_count == 0:
        # TODO: Check when happens
        false_prob = -1
    else:
        false_prob = float(false_seg_count) / float(total_count)

    return false_prob, total_count

def win_diff(h, gold, window_size=-1):
        """
        :param gold: gold segmentation (item in the list contains the number of words in segment) 
        :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
        :param window_size: optional 
        :return: accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.window_diff(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.window_diff(h, gold, return_parts=True)

        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)

        return false_prob, total_count