import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import utils.basic
from utils.basic import print_stats
import utils.samp
import utils.misc
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

def balanced_ce_loss(pred, gt):
    # pred and gt are the same shape
    for (a,b) in zip(pred.size(), gt.size()):
        assert(a==b) # some shape mismatch!
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
    
    pos_loss = utils.basic.reduce_masked_mean(loss, pos)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss, loss

def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert(D==2)
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert(S==S1)
    assert(S==S2)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i]#[:,:,0:1]
        i_loss = (flow_pred - flow_gt).abs() # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3) # B, S, N
        flow_loss += i_weight * utils.basic.reduce_masked_mean(i_loss, valids)
    flow_loss = flow_loss/n_predictions
    return flow_loss

def score_map_loss(fcps, trajs_g, vis_g, valids):
    #     # fcps is B,S,I,N,H8,W8
    B, S, I, N, H8, W8 = fcps.shape
    fcp_ = fcps.permute(0,1,3,2,4,5).reshape(B*S*N,I,H8,W8) # BSN,I,H8,W8
    # print('fcp_', fcp_.shape)
    xy_ = trajs_g.reshape(B*S*N,2).round().long() # BSN,2
    vis_ = vis_g.reshape(B*S*N) # BSN
    valid_ = valids.reshape(B*S*N) # BSN
    x_, y_ = xy_[:,0], xy_[:,1] # BSN
    ind = (x_ >= 0) & (x_ <= (W8-1)) & (y_ >= 0) & (y_ <= (H8-1)) & (valid_ > 0) & (vis_ > 0) # BSN
    fcp_ = fcp_[ind] # N_,I,H8,W8
    xy_ = xy_[ind] # N_
    N_ = fcp_.shape[0]
    # N_ is the number of heatmaps with valid targets

    # make gt with ones at the rounded spatial inds in here
    gt_ = torch.zeros_like(fcp_) # N_,I,H8,W8
    gt_[:,:,xy_[:,1],xy_[:,0]] = 1 # N_,I,H8,W8 with a 1 in the right spot

    ## softmax
    # fcp_ = fcp_.reshape(N_*I,H8*W8)
    # gt_ = gt_.reshape(N_*I,H8*W8)
    # argm = torch.argmax(gt_, dim=1)
    # ce_loss = F.cross_entropy(fcp_, argm, reduction='mean')

    ## ce
    fcp_ = fcp_.reshape(N_*I*H8*W8)
    gt_ = gt_.reshape(N_*I*H8*W8)
    # ce_loss = F.binary_cross_entropy_with_logits(fcp_, gt_, reduction='mean')
    ce_loss, _ = balanced_ce_loss(fcp_, gt_)
    # print('ce_loss', ce_loss)
    return ce_loss


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(S, input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = InstanceNormAlternative(planes)
            self.norm2 = InstanceNormAlternative(planes)
            if not stride == 1:
                self.norm3 = InstanceNormAlternative(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

# Ref: https://zenn.dev/pinto0309/scraps/1fa87e6fa6f918
class InstanceNormAlternative(nn.InstanceNorm2d):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(inp)
        desc = 1 / (inp.var(axis=[2, 3], keepdim=True, unbiased=False) + self.eps) ** 0.5
        retval = (inp - inp.mean(axis=[2, 3], keepdim=True)) * desc
        return retval

class GroupNormAlternative(nn.GroupNorm):

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(inp)

        desc = 1 / (input.var(axis=[2, 3], keepdim=True, unbiased=False) + self.eps) ** 0.5
        retval = (input - input.mean(axis=[2, 3], keepdim=True)) * desc
        return retval

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = 64
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim*2)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim*2)

        elif self.norm_fn == 'instance':
            self.norm1 = InstanceNormAlternative(self.in_planes)
            self.norm2 = InstanceNormAlternative(output_dim*2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64,  stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128+96+64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64,  stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(128+128+96+64, output_dim*2, kernel_size=3, padding=1, padding_mode='zeros')
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, InstanceNormAlternative, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        _, _, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)


        H8 = torch.div(H, self.stride, rounding_mode='trunc')
        W8 = torch.div(W, self.stride, rounding_mode='trunc')

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(a, (H8, W8), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H8, W8), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H8, W8), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a,b,c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(a, (H8, W8), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H8, W8), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H8, W8), mode='bilinear', align_corners=True)
            d = F.interpolate(d, (H8, W8), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a,b,c,d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
        
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x

class DeltaBlock(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3, S=8):
        super(DeltaBlock, self).__init__()
        
        self.input_dim = input_dim

        use_ones = True
        if use_ones:
            kitchen_dim = 2*(corr_levels * (2*corr_radius + 1)**2) + input_dim + 64*3 + 3
        else:
            kitchen_dim = (corr_levels * (2*corr_radius + 1)**2) + input_dim + 64*3 + 3

        self.hidden_dim = hidden_dim

        self.S = S
        
        self.to_delta = MLPMixer(
            S=self.S,
            input_dim=kitchen_dim,
            dim=512,
            output_dim=self.S*(input_dim+2),
            depth=12,
        )
            
        
    def forward(self, fhid, fcorr, flow):
        B, S, D = flow.shape
        assert (D == 3)

        flow_sincos = utils.misc.get_3d_embedding(flow, 64, cat_coords=True)
        x = torch.cat([fhid, fcorr, flow_sincos], dim=2) # B, S, -1
        delta = self.to_delta(x)
        delta = delta.reshape(B, self.S, self.input_dim+2)
        return delta

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        # print('fmaps', fmaps.shape)
        self.S, self.C, self.H, self.W = S, C, H, W

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        # print('fmaps', fmaps.shape)

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels-1):
            fmaps_ = fmaps.reshape(B*S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)
            # print('fmaps', fmaps.shape)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert(D==2)

        x0 = coords[:,0,:,0]
        y0 = coords[:,0,:,1]

        use_ones = True
        
        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i] # B, S, N, H, W
            if use_ones:
                ones = torch.ones_like(corrs)
            _, _, _, H, W = corrs.shape
            
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device) 

            centroid_lvl = coords.reshape(B*S*N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B*S*N, 1, H, W), coords_lvl)
            if use_ones:
                ones = bilinear_sampler(ones.reshape(B*S*N, 1, H, W), coords_lvl.detach()).detach()
            corrs = corrs.view(B, S, N, -1)
            if use_ones:
                ones = ones.view(B, S, N, -1)
                corrs = torch.cat([corrs, ones], dim=3) # B,S,N,RR*2
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1) # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert(C==self.C)
        assert(S==self.S)

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H*W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W) 
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)

class Pips(nn.Module):
    def __init__(self, S=8, stride=8, iters=3, with_feature = False):
        super(Pips, self).__init__()

        self.S = S
        self.stride = stride
        self.iters = iters
        self.with_feature = with_feature

        self.hidden_dim = hdim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        
        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        
        self.delta_block = DeltaBlock(input_dim=self.latent_dim, hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius, S=self.S)
        
        self.norm = nn.LayerNorm(self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

    def forward(self, xys, rgbs, feat_init=None):

        B, N, D = xys.shape
        assert(D==2)

        B, S, C, H, W = rgbs.shape

        rgbs = 2 * (rgbs / 255.0) - 1.0

        H8 = torch.div(H, self.stride, rounding_mode='trunc')
        W8 = torch.div(W, self.stride, rounding_mode='trunc')

        device = rgbs.device

        rgbs_ = rgbs.reshape(B*S, C, H, W)
        fmaps_ = self.fnet(rgbs_)
        fmaps = fmaps_.reshape(B, S, self.latent_dim, H8, W8)

        xys_ = xys.clone()/float(self.stride)

        coords = xys_.reshape(B, 1, N, 2).repeat(1, S, 1, 1) # init with zero vel

        hdim = self.hidden_dim

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        if self.with_feature and feat_init is not None:
            ffeat = feat_init
        else:
            # initialize features for the whole traj, using the initial feature
            ffeat = utils.samp.bilinear_sample2d(fmaps[:,0], coords[:,0,:,0], coords[:,0,:,1]).permute(0, 2, 1) # B, N, C

        ffeats = ffeat.reshape((B, 1, N, ffeat.size()[-1])).repeat(1, S, 1, 1) # B, S, N, C

        coord_predictions = []
        coords_bak = coords.clone()

        fcps = []
        ccps = []
        kps = []

        for itr in range(self.iters):
            coords = coords.detach()

            fcorr_fn.corr(ffeats)

            fcp = torch.zeros((B,S,N,H8,W8), dtype=torch.float32, device=device) # B,S,N,H8,W8
            for cr in range(self.corr_levels):
                fcp_ = fcorr_fn.corrs_pyramid[cr] # B,S,N,?,? (depending on scale)
                _,_,_,H_,W_ = fcp_.shape
                fcp_ = fcp_.reshape(B*S,N,H_,W_)
                fcp_ = F.interpolate(fcp_, (H8, W8), mode='bilinear', align_corners=True)
                fcp = fcp + fcp_.reshape(B,S,N,H8,W8)
            fcps.append(fcp)

            fcorrs = fcorr_fn.sample(coords) # B, S, N, LRR
            LRR = fcorrs.shape[3]

            # for mixer, i want everything in the format B*N, S, C
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B*N, S, LRR)
            flows_ = (coords - coords[:,0:1]).permute(0,2,1,3).reshape(B*N, S, 2)
            # coords_ = coords.permute(0,3,1,2) # B, 2, S, N
            times_ = torch.linspace(0, S, S, device=device).reshape(1, S, 1).repeat(B*N, 1, 1) # B*N,S,1
            flows_ = torch.cat([flows_, times_], dim=2) # B*N,S,2

            ffeats_ = ffeats.permute(0,2,1,3).reshape(B*N,S,self.latent_dim)
            
            delta_all_ = self.delta_block(ffeats_, fcorrs_, flows_) # B*N, S, C+2
            delta_coords_ = delta_all_[:,:,:2]
            delta_feats_ = delta_all_[:,:,2:]

            ffeats_ = ffeats_.reshape(B*N*S,self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B*N*S, self.latent_dim)
            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_
            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(0,2,1,3) # B,S,N,C

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3)

            coords[:,0] = coords_bak[:,0] # lock coord0 for target

            coord_predictions.append(coords * self.stride)

        vis_e = self.vis_predictor(ffeats.reshape(B*S*N, self.latent_dim)).reshape(B,S,N)

        fcps = torch.stack(fcps, dim=2) # B, S, I, N, H8, W8

        return coord_predictions[-1], vis_e, ffeat

if __name__ == '__main__':
    import onnx
    from onnxsim import simplify

    model = Pips(S=8, stride=4, iters=1, with_feature = False).cuda()
    model.eval()
    model_path = "../reference_model/model-000100000.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    B, S, C, H, W = 1, 8, 3, 480, 640
    N, D = 16*16, 2

    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
    grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
    xy = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2


    img = torch.randn((B, S, C, H, W), dtype=torch.float32, device='cuda')

    with torch.no_grad():
        coord_predictions, vis_e, ffeat = model(xy, img)
        # print(coord_predictions.shape)
        # print(vis_e.shape)
        # print(ffeat.shape)

        # Export the model
        onnx_path = "../pips.onnx"
        torch.onnx.export(model,
                          (xy, img),
                          onnx_path,
                          export_params=True,
                          opset_version=16,
                          do_constant_folding=True,
                          input_names=['xy','img'],
                          output_names=['coord_predictions', 'vis_e', 'ffeat'],
                          dynamic_axes={'xy': {1: 'num_points'},
                                        'coord_predictions': {2: 'num_points'},
                                        'vis_e': {2: 'num_points'},
                                        'ffeat': {1: 'num_points'}})

        # load your predefined ONNX model
        model = onnx.load(onnx_path)

        # convert model
        model_simp, check = simplify(model)

        # save the converted model
        onnx.save(model_simp, "../pips_simp.onnx")