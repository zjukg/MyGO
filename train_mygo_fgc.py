from dataset import VTKG
from model_mygo import MyGO
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random
import distutils
import logging

from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens

OMP_NUM_THREADS=8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(2024)
random.seed(2024)
np.random.seed(2024)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="MKG-W", type = str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--valid_epoch', default=50, type=int)
    parser.add_argument('--exp', default='mygo')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--hidden_dim', default=200, type = int)
    parser.add_argument('--dropout', default=0.01, type = float)
    parser.add_argument('--emb_dropout', default=0.9, type = float)
    parser.add_argument('--vis_dropout', default=0.4, type = float)
    parser.add_argument('--txt_dropout', default=0.1, type = float)
    parser.add_argument('--smoothing', default=0.0, type = float)
    parser.add_argument('--batch_size', default=2048, type = int)
    parser.add_argument('--decay', default=0.0, type = float)
    parser.add_argument('--max_img_num', default=3, type = int)
    parser.add_argument('--cont', action = 'store_true')
    parser.add_argument('--step_size', default=50, type = int)
    parser.add_argument('--max_vis_token', default=8, type=int)
    parser.add_argument('--max_txt_token', default=8, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)
    parser.add_argument('--mu', default=0, type=float)
    args = parser.parse_args()

    file_format = ""

    for arg_name in vars(args).keys():
        if arg_name in ["lr", "hidden_dim", "batch_size", "num_epoch", "max_vis_token", "max_txt_token", "num_head", "mu"]:
            file_format += f"{arg_name}_{vars(args)[arg_name]}"
        elif arg_name in ["score_function", "emb_dropout", "vis_dropout", "txt_dropout"]:
            file_format += f"{vars(args)[arg_name]}"


    if not args.no_write:
        os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok = True)
        os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok = True)
        os.makedirs(f"./logs/{args.exp}/{args.data}", exist_ok = True)
        if not os.path.isfile(f"ckpt/{args.exp}/args.txt"):
            with open(f"ckpt/{args.exp}/args.txt", "w") as f:
                for arg_name in vars(args).keys():
                    if arg_name not in ["data", "exp", "no_write", "num_epoch", "cont", "early_stop"]:
                        f.write(f"{arg_name}\t{type(vars(args)[arg_name])}\n")
    else:
        file_format = None

    file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.data}/{file_format}.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    logger.info(f"{os.getpid()}")
    logger.info(args)
    KG = VTKG(args.data, logger, max_vis_len = args.max_img_num)
    KG_Loader = torch.utils.data.DataLoader(KG, batch_size = args.batch_size, shuffle=True)
    visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    visual_token_index = visual_token_index.cuda()
    text_token_index, text_key_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)
    text_token_index = text_token_index.cuda()
    logger.info(visual_token_index, text_token_index)
    logger.info(visual_key_mask, text_key_mask)
    model = MyGO(
        num_ent = KG.num_ent, 
        num_rel = KG.num_rel,
        ent_vis_mask = visual_key_mask,
        ent_txt_mask = text_key_mask,
        dim_str = args.dim,
        num_head = args.num_head,
        dim_hid = args.hidden_dim,
        num_layer_enc_ent = args.num_layer_enc_ent,
        num_layer_enc_rel = args.num_layer_enc_rel,
        num_layer_dec = args.num_layer_dec,
        dropout = args.dropout,
        emb_dropout = args.emb_dropout,
        vis_dropout = args.vis_dropout, 
        txt_dropout = args.txt_dropout,
        visual_token_index = visual_token_index,
        text_token_index = text_token_index,
        score_function = args.score_function
    ).cuda()

    loss_fn = nn.CrossEntropyLoss(label_smoothing = args.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult = 2)

    last_epoch = 0
    start = time.time()
    logger.info("EPOCH\tLOSS\tTOTAL TIME")
    all_ents = torch.arange(KG.num_ent).cuda()
    all_rels = torch.arange(KG.num_rel).cuda()

    best_mrr = 0.0

    for epoch in range(last_epoch + 1, args.num_epoch + 1):
        total_loss = 0.0
        for batch, label in KG_Loader:
            ent_embs, rel_embs = model()
            scores = model.score(ent_embs, rel_embs, batch.cuda())
            loss = loss_fn(scores, label.cuda())
            if args.mu != 0:
                loss += model.contrastive_loss_finegrained(ent_embs) * args.mu
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        scheduler.step()
        logger.info(f"{epoch} \t {total_loss:.6f} \t {time.time() - start:.6f} s")
        if (epoch) % args.valid_epoch == 0:
            model.eval()
            with torch.no_grad():
                ent_embs, rel_embs = model()
                lp_list_rank = []
                for triplet in tqdm(KG.valid):
                    h,r,t = triplet
                    head_score = model.score(ent_embs, rel_embs, torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])

                    lp_list_rank.append(head_rank)
                    lp_list_rank.append(tail_rank)

                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Validation Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

                lp_list_rank = []
                for triplet in tqdm(KG.test):
                    h,r,t = triplet
                    head_score = model.score(ent_embs, rel_embs, torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).cuda())[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])

                    lp_list_rank.append(head_rank)
                    lp_list_rank.append(tail_rank)

                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Test Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

            if best_mrr < mrr:
                best_mrr = mrr
                best_result = (mr, mrr, hit10, hit3, hit1)

            model.train()
            if (epoch) % 500 == 0:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    },
                    f"./ckpt/{args.exp}/{args.data}/{file_format}_{epoch}.ckpt"
                )

            model.train()
    
    logger.info("Done! {}. The best results are shown below:".format(args.data))
    logger.info(best_result)
