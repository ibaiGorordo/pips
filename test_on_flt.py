import time
import argparse
import numpy as np
import timeit
import matplotlib
# import tensorflow as tf
# import scipy.misc
import io
import os
import math
from PIL import Image
matplotlib.use('Agg') # suppress plot showing

import matplotlib.pyplot as plt

import matplotlib.animation as animation
import cv2
import saverloader

#from raft_core.raft import RAFT
# from relative_perceiver import RelativePerceiver
# from sparse_relative_perceiver import SparseRelativePerceiver
# from nets.graph_raft import GraphRaft
#from nets.st_graph_raft import StGraphRaft
# from nets.st_spraft import StSpRaft
# from nets.st_graph_raft import StGraphRaft
# # from nets.mraft import Mraft
# from nets.praft import Praft
# from nets.mpraft import Mpraft
# from perceiver_graph import PerceiverGraph
# from relative_mlp import RelativeMlp
# import nets.raftnet
from nets.raftnet import Raftnet
from nets.singlepoint import Singlepoint

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

import utils.py
# import utils.box
import utils.misc
import utils.improc
# import utils.vox
# import utils.grouping
# from tqdm import tqdm
import random
# import glob
# import color2d

from utils.basic import print_, print_stats


# import relation_model

import flyingthingsdataset
# import datasets
# import cater_pointtraj_dataset
from headtrackingdataset import HeadTrackingDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

# import inputs

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

def prep_frame_for_dino(img, scale_size=[192]):
    """
    read a single frame & preprocess
    """
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()

    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x
    
    img = color_normalize(img)
    return img, ori_h, ori_w

def get_feats_from_dino(model, frame):
    # batch version of the other func
    B = frame.shape[0]
    h, w = int(frame.shape[2] / model.patch_embed.patch_size), int(frame.shape[3] / model.patch_embed.patch_size)
    out = model.get_intermediate_layers(frame.cuda(), n=1)[0] # B, 1+h*w, dim
    dim = out.shape[-1]
    out = out[:, 1:, :]  # discard the [CLS] token
    outmap = out.permute(0, 2, 1).reshape(B, dim, h, w)
    return out, outmap, h, w

def restrict_neighborhood(h, w):
    size_mask_neighborhood = 12
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)

def label_propagation(h, w, feat_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    # print('feat_tar', feat_tar.shape)
    # print('feat_sources', feat_sources.shape)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)

    size_mask_neighborhood = 12
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h*w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    topk = 5
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    
    return seg_tar, mask_neighborhood

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def run_dino(dino, d, sw):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    import copy

    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S1, N, D = trajs_g.shape

    sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)

    # trajs_g = trajs.clone()
    # vis_g = vis.clone()
    # valids = valid.clone()
    
    # all_visible = (vis_g.sum(1) == S).reshape(-1)

    _, S, C, H, W = rgbs.shape

    patch_size = 8

    assert(B==1)
    xy0 = trajs_g[:,0] # B, N, 2

    # The queue stores the n preceeding frames
    import queue
    import copy
    n_last_frames = 7
    que = queue.Queue(n_last_frames)

    # run dino
    prep_rgbs = []
    for s in range(S):
        prep_rgb, ori_h, ori_w = prep_frame_for_dino(rgbs[0, s].permute(1,2,0).detach().cpu().numpy(), scale_size=[H])
        prep_rgbs.append(prep_rgb)
    prep_rgbs = torch.stack(prep_rgbs, dim=0) # S, 3, H, W
    with torch.no_grad():
        bs = 8
        idx = 0 
        featmaps = []
        while idx < S:
            end_id = min(S, idx+bs)
            _, featmaps_cur, h, w = get_feats_from_dino(dino, prep_rgbs[idx:end_id]) # S, C, h, w
            idx = end_id
            featmaps.append(featmaps_cur)
        featmaps = torch.cat(featmaps, dim=0)
    C = featmaps.shape[1]
    featmaps = featmaps.unsqueeze(0) # 1, S, C, h, w
    # featmaps = F.normalize(featmaps, dim=2, p=2)

    xy0 = trajs_g[:, 0, :] # B, N, 2
    first_seg = torch.zeros((1, N, H//patch_size, W//patch_size))
    for n in range(N):
        first_seg[0, n, (xy0[0, n, 1]/patch_size).long(), (xy0[0, n, 0]/patch_size).long()] = 1

    frame1_feat = featmaps[0, 0].reshape(C, h*w) # dim x h*w
    mask_neighborhood = None
    accs = []
    trajs_e = torch.zeros_like(trajs_g).to(device)
    trajs_e[0,0] = trajs_g[0,0]
    for cnt in range(1, S):
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        feat_tar = featmaps[0, cnt].reshape(C, h*w)

        frame_tar_avg, mask_neighborhood = label_propagation(h, w, feat_tar.T, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        for n in range(N):
            vis = vis_g[0,cnt,n]
            if len(torch.nonzero(frame_tar_avg[n])) > 0:
                # weighted average
                nz = torch.nonzero(frame_tar_avg[n])
                coord_e = torch.sum(frame_tar_avg[n][nz[:,0], nz[:,1]].reshape(-1,1) * nz.float(), 0) / frame_tar_avg[n][nz[:,0], nz[:,1]].sum() # 2
                coord_e = coord_e[[1,0]]
            else:
                # stay where it was
                coord_e = trajs_e[0,cnt-1,n]
                
            trajs_e[0, cnt, n] = coord_e
            # if vis > 0:
            #     coord_g = trajs_g[0,cnt,n] # 2
            #     dist = torch.sqrt(torch.sum((coord_e-coord_g)**2, dim=0))
            #     # print_('dist', dist)
            #     area = torch.sum(segs[0,cnt])
            #     # print_('0.2*sqrt(area)', 0.2*torch.sqrt(area))
            #     thr = 0.2 * torch.sqrt(area)
            #     correct = (dist < thr).float()
            #     accs.append(correct)

        # if len(accs) > 0:
        #     print(torch.mean(torch.stack(accs)) * 100.0)

    # pck = torch.mean(torch.stack(accs)) * 100.0
    # metrics['pck'] = pck.item()

    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))

        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)
    
    return total_loss, metrics


def run_singlepoint(model, d, sw):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)

    # sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_black', trajs_g, torch.ones_like(rgbs)*-0.5)
    sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)

    # trajs_g = trajs.clone()
    # vis_g = vis.clone()
    # valids = valid.clone()
    
    # all_visible = (vis_g.sum(1) == S).reshape(-1)

    _, S, C, H, W = rgbs.shape

    # if sw is not None and sw.save_this:
    #     sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))

    # preds, preds2, cps, vis_e = model(trajs_g[:,0], rgbs, trajs_g=trajs_g, vis_g=vis_g, valids=valids, iters=6, sw=sw)
    preds, preds2, vis_e, stats = model(trajs_g[:,0], rgbs, iters=6, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    
    seq_loss, vis_loss, ce_loss = stats

    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    trajs_e = preds[-1]

    if sw is not None and sw.save_this:
        
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)

        # animate_traj2ds_on_rgbs
        # rgb_vis = []
        # black_vis = []
        # for trajs_e in preds2:
        #     trajs_e = trajs_e.reshape(B_bak, N2_bak, S, 2).permute(0, 2, 1, 3) + pad
        #     rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_rgb, only_return=True, cmap='coolwarm'))
        #     black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_black, only_return=True, cmap='coolwarm'))
        # sw.summ_rgbs('outputs/animated_trajs_on_black', black_vis)
        # sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return total_loss, metrics


def run_raft(raft, d, sw):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)

    # sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_black', trajs_g, torch.ones_like(rgbs)*-0.5)
    sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)

    # trajs_g = trajs.clone()
    # vis_g = vis.clone()
    # valids = valid.clone()
    
    # all_visible = (vis_g.sum(1) == S).reshape(-1)

    _, S, C, H, W = rgbs.shape

    prep_rgbs = utils.improc.preprocess_color(rgbs)

    flows_e = []
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        rgb1 = prep_rgbs[:,s+1]
        flow, _ = raft(rgb0, rgb1, iters=32)
        flows_e.append(flow)
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    coord0 = trajs_g[:,0] # B, N, 2
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2
    
    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))

        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)

    return total_loss, metrics


def train(args):

    # default coeffs (don't touch)
    init_dir = ''
    coeff_prob = 0.0
    use_augs = False

    exp_name = 'fa00' # copy from particles repo
    
    # init_dir = '/projects/katefgroup/track_share/12_8_256_8_1e-4_p1_A_big16_19:31:10'
    init_dir = 'saved_checkpoints/4hv_8_128_I6_4e-4_p1_A_tb78_01:13:39' # best
    init_dir = 'checkpoints/4hv_8_128_I6_4e-4_p1_A_tb79_09:11:39' # new best
    
    stride = 4

    ## choose hyps
    B = 1
    S = 8
    N = 16
    lr = 1e-4
    grad_acc = 1

    S_stride = 3

    max_iters = 1000
    log_freq = 100
    save_freq = 100000
    shuffle = True
    do_val = False
    val_every_step = True
    subset = 'all'
    crop_size = (384,512) # the raw data is 540,960

    # actual coeffs
    coeff_prob = 1.0

    use_augs = True
    use_augs = False

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    all_coeffs = [
        coeff_prob,
    ]
    all_prefixes = [
        "p",
    ]
    for l_, l in enumerate(all_coeffs):
        if l > 0:
            model_name += "_%s%s" % (all_prefixes[l_], utils.basic.strnum(l))
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name

    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_test_on_flt'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TEST', subset=subset,
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=12,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    train_iterloader = iter(train_dataloader)

    global_step = 0
    
    raft = Raftnet(ckpt_name='../RAFT/models/raft-things.pth').cuda()
    raft.eval()

    singlepoint = Singlepoint(S=S, stride=stride).cuda()
    if init_dir:
       _ = saverloader.load(init_dir, singlepoint)
    singlepoint.eval()

    patch_size = 8
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size).cuda()
    for p in dino.parameters():
        p.requires_grad = False
    dino.eval()
    

    n_pool = 10000
    # loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # epe_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # epe_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # epe_inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # epe_inv2inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    # flow_pool_t = utils.misc.SimplePool(n_pool, version='np')
    
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = utils.misc.SimplePool(n_pool, version='np')

    max_iters = len(train_dataloader)
    print('setting max_iters', max_iters)
    
    while global_step < max_iters:
        # optimizer.zero_grad()
        
        read_start_time = time.time()

        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        try:
            sample = next(train_iterloader)
        except StopIteration:
            train_iterloader = iter(train_dataloader)
            sample = next(train_iterloader)

        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        with torch.no_grad():
            # total_loss, metrics = run_singlepoint(singlepoint, sample, sw_t)
            # total_loss, metrics = run_raft(raft, sample, sw_t)
            total_loss, metrics = run_dino(dino, sample, sw_t)

        # sw_t.summ_scalar('total_loss', total_loss)
        # loss_pool_t.update([total_loss.detach().cpu().numpy()])
        # sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())

        # if np.mod(global_step, save_freq)==0:
        #     saverloader.save(ckpt_dir, optimizer, model, global_step, keep_latest=5)

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f, ate_vis = %.2f, ate_occ = %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            ate_vis_pool_t.mean(), ate_occ_pool_t.mean()))
        # if global_step % log_freq == 0:
        #     print('epe_pooled = %.5f, epe_vis_pooled = %.5f, epe_inv_pooled = %.5f' % (
        #         epe_pool_t.mean(), epe_vis_pool_t.mean(), epe_inv_pool_t.mean()))

    writer_t.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='raft', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training") 
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--validation', type=str, nargs='+')
    # parser.add_argument('--lr', type=float, default=0.00002)
    # parser.add_argument('--num_steps', type=int, default=100)
    # parser.add_argument('--num_steps', type=int, default=10000)
    # parser.add_argument('--num_steps', type=int, default=300000)
    # parser.add_argument('--num_steps', type=int, default=300000)
    # parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    # parser.add_argument('--lr', type=float, default=4e-4)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--lr', type=float, default=0.00025)
    # parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    
    train(args)

