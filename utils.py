import numpy as np

def calculate_rank(score, target, filter_list):
	score_target = score[target]
	score[filter_list] = score_target - 1
	rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
	return rank

def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1