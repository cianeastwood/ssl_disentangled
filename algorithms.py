"""Contrastive learning algorithms. """

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import off_diagonal, FullGatherLayer, accuracy, extract_interleaved_views
EPS = 1e-6


def get_algorithm_class(alg_name):
    """Return the algorithm class with the given name."""
    name_class_dict = {k.lower(): v for k, v in globals().items()}
    if alg_name not in name_class_dict:
        raise NotImplementedError("Algorithm not found: {}".format(alg_name))
    return name_class_dict[alg_name]


#############################################################################################
######################## VANILLA METHODS ####################################################
#############################################################################################

class BaseCL(nn.Module):
    """
    Base class for all contrastive-learning algorithms.
    """
    def __init__(self, args, backbone, projector, projector_style=None):
        super().__init__()
        self.args = args
        self.z_size = int(args.mlp.split("-")[-1])
        self.backbone = backbone
        self.projector = projector
        self.classifier_h = nn.Linear(args.h_size, args.n_classes)
        self.classifier_z = nn.Linear(self.z_size, args.n_classes)
        self.lambda_default = 1.0

        # Style projector
        self.projector_style = projector_style
        if args.mlp_style is not None:
            self.z_size_style = int(args.mlp_style.split("-")[-1])
            self.classifier_z_style = nn.Linear(self.z_size_style, args.n_classes)
            self.classifier_z_joint = nn.Linear(self.z_size + self.z_size_style, args.n_classes)
    
    def forward_networks(self, views):
        # Gather h, z and corresponding logits for the first view (used for stats/accuracies)
        h0 = self.backbone(views[0])
        z0 = self.projector(h0)
        zs = [z0]
        logits_h = self.classifier_h(h0.detach())
        logits_z = self.classifier_z(z0.detach())
        logits_z_style, logits_z_joint = None, None

        # Gather for style projector
        if self.projector_style is not None:
            z0_style = self.projector_style(h0)
            z0_joint = torch.cat([z0, z0_style], dim=1)
            zs = [z0_joint]
            logits_z_style = self.classifier_z_style(z0_style.detach())
            logits_z_joint = self.classifier_z_joint(z0_joint.detach())

        # Gather zs for the remaining views (used for main losses)
        for view in views[1:]:
            h_v = self.backbone(view)
            z_v = self.projector(h_v)
            if self.projector_style is not None:
                z_v_style = self.projector_style(h_v)
                z_v = torch.cat([z_v, z_v_style], dim=1)
            zs.append(z_v)

        return zs, logits_h, logits_z, logits_z_style, logits_z_joint
            

    def online_classification(self, logits_h, logits_z, labels, 
                              logits_z_style=None, logits_z_joint=None):
        # Setup
        all_logits = [logits_h, logits_z]
        all_logits_names = ["h", "z"]
        if logits_z_style is not None:
            all_logits.append(logits_z_style)
            all_logits_names.append("z_s")
        if logits_z_joint is not None:
            all_logits.append(logits_z_joint)
            all_logits_names.append("z_j")

        # Compute losses and accuracies
        classif_losses = 0.
        classif_stats = {}
        for logits, logits_name in zip(all_logits, all_logits_names):
            classif_loss = F.cross_entropy(logits, labels)
            if self.args.n_classes > 5:
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            else:
                [acc1] = accuracy(logits, labels, topk=(1,))
                acc5 = torch.ones(1, device=logits.device)
            classif_stats[f"classif_loss_{logits_name}"] = classif_loss
            classif_stats[f"acc1_{logits_name}"] = acc1
            classif_stats[f"acc5_{logits_name}"] = acc5
            classif_losses = classif_losses + classif_loss

        return classif_losses, classif_stats

    def forward(self, inputs_dict, lambd=None, is_val=False):
        lambd = self.lambda_default if lambd is None else lambd
        zs, logits_h, logits_z, logits_z_style, logits_z_joint = self.forward_networks(inputs_dict["views"])
        del inputs_dict["views"]
        
        # Training loss
        if is_val:
            # Don't compute training loss for validation (not meaningful as no augmentations are used)
            loss = 0.
            logs = {}
        else:
            loss, logs = self.loss(zs, lambd)
        
        # Online classification
        classif_losses, classif_stats = self.online_classification(logits_h, logits_z, inputs_dict["labels"],
                                                                   logits_z_style, logits_z_joint)
        loss = loss + classif_losses    # ensure classifier is updated (z.detach() blocks grads from labels)
        logs.update(classif_stats)

        return loss, logs

    def loss(self, zs, lambd):
        raise NotImplementedError("Must be implemented by subclass")
    
    def inv_constraint_loss(self, inv_constraint_value, *args, **kwargs):
        return inv_constraint_value


class SimCLR(BaseCL):
    """
    SimCLR algorithm.
    Based on: https://github.com/facebookresearch/vissl/blob/main/vissl/losses/simclr_info_nce_loss.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_views = self.args.n_views
        self.precompute_pos_neg_mask(self.n_views)
        
    def precompute_pos_neg_mask(self, n_pos=2, use_gpu=True):
        """
        Precompute the positive and negative masks to speed up the loss calculation.
        """
        # Extract params
        total_images = self.args.batch_size * n_pos
        batch_size_gpu = total_images // self.args.world_size
        orig_images_gpu = batch_size_gpu // n_pos
        rank = self.args.rank
        
        # More setup
        pos_mask = torch.zeros(batch_size_gpu, total_images)
        neg_mask = torch.zeros(batch_size_gpu, total_images)
        all_indices = np.arange(total_images)
        pos_members = orig_images_gpu * np.arange(n_pos)
        orig_members = torch.arange(orig_images_gpu)

        # Build masks
        for anchor in np.arange(n_pos):
            for img_idx in range(orig_images_gpu):
                delete_inds = batch_size_gpu * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * orig_images_gpu + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(n_pos), anchor):
                pos_inds = batch_size_gpu * rank + pos * orig_images_gpu + orig_members
                pos_mask[
                    torch.arange(
                        anchor * orig_images_gpu, (anchor + 1) * orig_images_gpu
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask.cuda(non_blocking=True) if use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) if use_gpu else neg_mask
    
    def loss(self, zs, lambd):
        assert len(zs) == self.n_views, f"Expected {self.n_views} positives/views, got {len(zs)}"

        # Get MSE (before normalization) for logging between query and each key view (positives)
        with torch.no_grad():
            mse_losses = [F.mse_loss(zs[0], zs[i]) for i in range(1, len(zs))]

        # Concat zs into a single tensor: [n_pos, batch_size, z_dim] --> [n_pos*batch_size, z_dim]
        zs = torch.concat(zs, dim=0)
        
        # Normalize 
        zs = F.normalize(zs, dim=1)

        # Gather
        if self.args.world_size > 1:
            zs_dist = torch.cat(FullGatherLayer.apply(zs), dim=0)
        else:
            zs_dist = zs
        
        # Matrix multiply: [n_pos*batch_size, z_dim] with [n_pos*batch_size*world_size, z_dim] gives:
        # [n_pos*batch_size, n_pos*batch_size*world_size]
        similarity = torch.exp(torch.mm(zs, zs_dist.t()) / self.args.temperature)   # may need a .contiguous() here
        
        # Negative loss
        neg_loss = torch.mean(torch.log(torch.sum(similarity * self.neg_mask, 1)))
        
        # Positive loss (numerator)
        pos = -torch.log(torch.sum(similarity * self.pos_mask, 1))
        if self.args.exclude_fract > 0.:
            pos = pos.sort()[0][:-int(self.args.exclude_fract * pos.shape[0])] # Exclude worst a-fraction
        pos_loss = torch.mean(pos)

        # Final loss
        loss = lambd * pos_loss + neg_loss
        
        # Logs
        logs = dict(loss=loss.detach().clone(), inv_constraint=pos_loss.detach(), entropy_term=neg_loss.detach())
        if len(mse_losses) == 1:
            logs.update(dict(mse_unnorm=mse_losses[0].detach()))
        else:
            logs.update({f"mse_unnorm_{i}": mse_loss.detach() for i, mse_loss in enumerate(mse_losses)})

        return loss, logs

    def inv_constraint_loss(self, inv_constraint_value, *args, **kwargs):
        """
        Standardize or normalise the invariant constraint term/loss (positive term for SimCLR).

        To do so, remove the effect of:
        - 1) summing over > 1 key view, where n_key_views=n_views-1; and
        - 2) temperature.
        
        Also add 1 to make it a positive loss (min=0).
        """
        return (inv_constraint_value - math.log(self.n_views - 1)) * self.args.temperature + 1.


class BarlowTwins(BaseCL):
    """
    Barlow Twins algorithm.

    Based on:
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.z_size, affine=False) # norm layer for z1 and z2

    def loss(self, zs, lambd):
        # Assumes 2 views: query- and key-view embeddings
        z1, z2 = zs
            
        # Logging only: get mse_loss
        with torch.no_grad():
            mse_loss = F.mse_loss(z1, z2)

        # Standardize each dim to have zero mean and unit variance
        N = z1.size()[0]
        z1, z2 = self.bn(z1), self.bn(z2)

        # Empirical cross-correlation matrix
        corr = z1.T @ z2 / N

        # Sum the cross-correlation matrix between all gpus
        if self.args.world_size > 1:
            torch.distributed.all_reduce(corr)
            corr /= self.args.world_size

        # Get diagonal/positive and off-diagonal/negative losses
        off_diag = off_diagonal(corr).pow_(2).sum()
        on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
        if self.args.exclude_fract > 0.:
            raise NotImplementedError("Exclude fraction not implemented for BarlowTwins")
        
        # Final loss (BTs default: lambd=1)
        loss = lambd * on_diag + self.args.lambd_bts * off_diag
        logs = dict(loss=loss.detach().clone(), mse_unnorm=mse_loss.detach(), 
                    inv_constraint=on_diag.detach(), entropy_term=off_diag.detach())

        return loss, logs
    
    def inv_constraint_loss(self, inv_constraint_value, *args, **kwargs):
        """
        Standardize or normalise the invariant constraint term/loss (on-diag term for BTs).

        To do so, remove the effect of:
        - 1) z-space dimensionality (post-projector), since a sum rather than mean is used

        Also has the benefit of ensuring the contraint scale is similar to other 
        algorithms, allowing a similar tolerance.
        """
        return inv_constraint_value / float(self.args.mlp.split("-")[-1])


class VICReg(BaseCL):
    """
    VICReg algorithm.
    
    Based on:
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py and
    https://github.com/facebookresearch/VICRegL/blob/main/main_vicregl.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_default = self.args.sim_coeff
    
    def loss(self, zs, lambd):
        # Assumes 2 views: query- and key-view embeddings
        z1, z2 = zs

        # Get inv_constraint / i.e. mse_loss / i.e. repr_loss
        if self.args.exclude_fract > 0.:
            # Exclude worst a-fraction
            per_sample_mses = F.mse_loss(z1, z2, reduction="none").mean(dim=1)
            mse_loss = per_sample_mses.sort()[0][:-int(self.args.exclude_fract * per_sample_mses.shape[0])].mean()
        else:
            mse_loss = F.mse_loss(z1, z2)

        # Gather across GPUS
        z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
        z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

        # Center
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # Calculate std loss
        std_x = torch.sqrt(z1.var(dim=0).type(torch.float32) + 0.0001)
        std_y = torch.sqrt(z2.var(dim=0).type(torch.float32) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Calculate cov loss
        cov_x = (z1.T @ z1) / (z1.size(0) - 1)
        cov_y = (z2.T @ z2) / (z2.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.z_size) / 2 + \
        off_diagonal(cov_y).pow_(2).sum().div(self.z_size) / 2

        # Final loss
        entropy_term = self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss
        loss = lambd * mse_loss + entropy_term

        logs = dict(loss=loss.detach().clone(), mse_unnorm=mse_loss.detach(), 
                    inv_constraint=mse_loss.detach(), entropy_term=entropy_term.detach(), 
                    std_loss=std_loss.detach(), cov_loss=cov_loss.detach())

        return loss, logs


######################################################################################################
######################## OUR MULTIHEAD ALGORITHMS ####################################################
######################################################################################################

class MultiHeadCL(BaseCL):
    """
    Base class for multi-head or multi-embedding-space contrastive-learning algorithms.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_spaces = self.n_key_views = max(1, len(self.args.z_split_fracts))     # number of embedding spaces and key views
        self.n_views = self.n_key_views + 1                                     # +1 for query view
        
        if self.projector_style is not None:
            # self.projector output used for content and self.projector_style output split into style spaces using z_split_fracs[1:] 
            self.z_split_inds = self.precompute_z_split_inds_sep(self.z_size, self.z_size_style, self.args.z_split_fracts)
        else:
            # self.projector output split into content and style spaces using z_split_fracs
            self.z_split_inds = self.precompute_z_split_inds(self.z_size, self.args.z_split_fracts)
        
        if self.args.eff_mv_dl:          # efficient multi-view data loading
            self.shared_q = False   # one query view per key view
        else:
            self.shared_q = True    # shared query view across key views
    
    def precompute_z_split_inds(self, z_size, split_fracts):
        """ Split z into multiple embedding spaces based on indices."""
        n_spaces = len(split_fracts)
        z_splits_sizes = [int(z_size * fract) for fract in split_fracts]
        z_splits_sizes[0] = z_splits_sizes[0] + (z_size - sum(z_splits_sizes))  # ensure sum of splits is z_size
        z_split_inds = [0] + [sum(z_splits_sizes[:i]) for i in range(1, n_spaces + 1)]  # [0, end_i, ...]
        z_split_inds = [np.arange(start=z_split_inds[i], stop=z_split_inds[i+1]) 
                        for i in range(n_spaces)]
        return z_split_inds
    
    def precompute_z_split_inds_sep(self, z_size, z_size_style, split_fracts):
        # Get style-space split indices
        style_fracs = [f * 1./sum(split_fracts[1:]) for f in split_fracts[1:]]   # renormalize
        assert sum(style_fracs) == 1.0
        style_split_inds = self.precompute_z_split_inds(z_size_style, style_fracs)
        style_split_inds = [inds + z_size for inds in style_split_inds]          # shift indices to style space

        # Get content-space split indices
        content_inds = np.arange(z_size)

        return [content_inds] + style_split_inds

    def _prep_lambd_arg(self, lambd):
        "Prepares lambd (List/Tensor) argument for forward pass."
        if lambd is None or len(lambd) == 0:
            lambd = [self.lambda_default] * self.n_spaces
        else:
            if isinstance(lambd, list):
                if len(lambd) == 1:
                    lambd = lambd * self.n_spaces
            elif isinstance(lambd, torch.Tensor):
                if len(lambd) == 1:
                    lambd = lambd.repeat(self.n_spaces)
            elif isinstance(lambd, float):
                lambd = [lambd] * self.n_spaces
            else:
                raise ValueError(f"lambd must be a float, list of floats, or tensor. Found {type(lambd)}")
        assert len(lambd) == self.n_spaces, "If provided, must have one lambda per space."
        return lambd
    
    def forward(self, inputs_dict, lambd=None, is_val=False):
        lambd = self._prep_lambd_arg(lambd)
        return super().forward(inputs_dict, lambd, is_val)
    
    def mse_loss_zspace_i(self, sq_errors_all_spaces, space_idx, exclude_fract=0.):
        sq_errors_space_i = sq_errors_all_spaces[:, self.z_split_inds[space_idx]]
        if exclude_fract > 0.:
            sq_errors_space_i = sq_errors_space_i.sort()[0][:-int(exclude_fract * len(sq_errors_space_i))]
        return torch.mean(sq_errors_space_i)

    def get_query_and_keys(self, zs):
        if self.args.eff_mv_dl:
            # Multiple query views (one per key view), with views in zs interleaved rather than grouped/stacked.
            z_qs = extract_interleaved_views(zs[0], self.n_key_views)                      # [n_views, batch_size_gpu, z_dim]
            z_ks = extract_interleaved_views(zs[1], self.n_key_views)                      # [n_views, batch_size_gpu, z_dim]
        else:
            # Single query and multiple key views, already grouped/stacked by view
            return zs[0], zs[1:]
        return z_qs, z_ks
    
    def get_mse_losses(self, z_qs, z_ks):
        mse_losses = []
        for i, z_k_i in enumerate(z_ks):
            z_q_i = z_qs if self.shared_q else z_qs[i]           
            sq_errors = F.mse_loss(z_q_i, z_k_i, reduction="none")
            mse_losses.append(self.mse_loss_zspace_i(sq_errors, i, self.args.exclude_fract))
            if self.args.m_cont_views and i > 0:     # Use all key-views for content space
                mse_losses[0] = mse_losses[0] + self.mse_loss_zspace_i(sq_errors, 0, self.args.exclude_fract)
        
        if self.args.m_cont_views:
            mse_losses[0] = mse_losses[0] / len(z_ks)   # average over all key views in content space

        return mse_losses


class SimCLRMultiHead(MultiHeadCL, SimCLR):
    """
    SimCLR multi-head algorithm (ours).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if efficient multi-view data loading, then negative mask compares across key views (different examples)
        n_pos = 2 if self.args.eff_mv_dl else self.args.n_views
        self.precompute_pos_neg_mask(n_pos=n_pos)
    
    def normalize_per_zspace(self, zs, p=2):
        """ 
        Normalize each embedding space *separately* to have length 1 (along the first dimension).
        Joint space thus has a norm == sqrt(n_spaces).
        """
        zs_out = torch.empty_like(zs)           # avoid in-place ops
        for i in range(self.n_spaces):
            zs_out[:, self.z_split_inds[i]] = F.normalize(zs[:, self.z_split_inds[i]], p=p, dim=1).type(zs.dtype)
        return zs_out

    def pos_loss_space_i(self, z_q, z_k, space_idx):
        z_q_space_s = z_q[:, self.z_split_inds[space_idx]]                      # [batch_size_gpu, z_dim_space_i]
        z_ki_space_s = z_k[:, self.z_split_inds[space_idx]]                     # [batch_size_gpu, z_dim_space_i]
        pos = -torch.sum(z_q_space_s * z_ki_space_s, dim=-1) / self.args.temperature  # [batch_size_gpu]
        if self.args.exclude_fract > 0.:        # exclude worst a-fraction
            pos = pos.sort()[0][:-int(self.args.exclude_fract * pos.size()[0])]
        return torch.mean(pos)

    def get_pos_losses(self, z_qs, z_ks):
        pos_losses = []
        for i, z_k_i in enumerate(z_ks):
            # Get query i if not shared_q, i.e., if we have a different query per key (using the efficient dataloader)
            z_q_i = z_qs if self.shared_q else z_qs[i]  

            # Compare query and key i in space i
            pos_losses.append(self.pos_loss_space_i(z_q_i, z_k_i, space_idx=i))  

            # (Optionally) Compare query and key i>0 in space 0 (content space)       
            if self.args.m_cont_views and i > 0:         # I.e., use all key-views for content space
                pos_losses[0] = pos_losses[0] + self.pos_loss_space_i(z_q_i, z_k_i, space_idx=0)

        # Average, if multiple key-views used in content space
        if self.args.m_cont_views:
            pos_losses[0] = pos_losses[0] / len(z_ks)

        return pos_losses
    
    def loss(self, zs, lambd):
        # Extract query and keys: single query, multiple keys
        z_q, z_ks = self.get_query_and_keys(zs)         # z_q: [batch_size_gpu, z_dim]
        
        # Logging only: get MSE (before normalization) for each space
        mse_losses = []
        with torch.no_grad():
            mse_losses = self.get_mse_losses(z_q, z_ks)

        # Normalize each space separately, joint norm = sqrt(n_spaces)
        zs = [self.normalize_per_zspace(z) for z in zs]
        z_q, z_ks = self.get_query_and_keys(zs) 

        # Positive loss: deal with each key-view separately, getting positive loss in each space
        pos_losses = self.get_pos_losses(z_q, z_ks)
        pos_loss = sum([lmd * pos_loss for lmd, pos_loss in zip(lambd, pos_losses)]) / self.n_spaces    # over all spaces
        
        # Negative loss
        # -- concat, normalize and gather
        zs = torch.concat(zs, dim=0)        # [n_views, batch_size_gpu, z_dim] --> [n_views*batch_size_gpu, z_dim]
        zs = zs / math.sqrt(self.n_spaces)  # normalize joint embeddings to have norm 1 (joint norm was =sqrt(n_spaces))
        if self.args.world_size > 1:
            zs_dist = torch.cat(FullGatherLayer.apply(zs), dim=0) # [n_views*batch_size_gpu*world_size, z_dim]
        else:
            zs_dist = zs
        # -- similarity matrix: # [n_views*batch_size_gpu, n_views*batch_size_gpu*world_size]
        sim = torch.exp(torch.mm(zs, zs_dist.t()) / self.args.temperature)
        # -- mask and sum: negatives are other instances, as usual
        neg_loss = torch.mean(torch.log(torch.sum(sim * self.neg_mask, dim=1)))
        # -- additional content-space negative loss
        if self.args.add_cont_neg:
            zs_cont_space = zs[:, self.z_split_inds[0]]                                 # [batch_size_gpu, z_dim_space_0]
            zs_dist_cont_space = zs_dist[:, self.z_split_inds[0]]                       # [batch_size_gpu, z_dim_space_0]
            sim_content_space = torch.exp(torch.mm(zs_cont_space, zs_dist_cont_space.t()) / self.args.temperature)
            neg_loss = neg_loss + torch.mean(torch.log(torch.sum(sim_content_space * self.neg_mask, dim=1)))
        
        # Final loss (lambdas are already applied to pos_losses above)
        loss = pos_loss + neg_loss
        
        # Logs
        logs = dict(loss=loss.detach().clone(), entropy_term=neg_loss.detach())
        logs.update({f"mse_loss_{i}": mse_loss.detach() for i, mse_loss in enumerate(mse_losses)})
        logs.update({f"inv_constraint_{i}": pos_loss.detach() for i, pos_loss in enumerate(pos_losses)})

        return loss, logs
    
    def inv_constraint_loss(self, inv_constraint_value, *args, **kwargs):
        """
        Standardize or normalise the invariant constraint term/loss.
        
        Content space: take mean over key views.
        Style space: compare only a single key-view with the query.
        
        Result: 
         - don't need to remove the effect of using multiple key views per instance.
         - only need to remove the effect of temperature, and add 1 to make it a positive loss (min=0).
        """
        return inv_constraint_value * self.args.temperature + 1.


class VICRegMultiHead(MultiHeadCL, VICReg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_default = self.args.sim_coeff

    def split_z_into_views(self, zs):
        """
        Split concatenated zs (back) into views.

        Input: Tensor of shape [n_views*batch_size*world_size, z_dim]
        Output: Iterable of n_views tensors of shape [batch_size*world_size, z_dim]
        """
        n_views = 2 if self.args.eff_mv_dl else self.n_views    # more negatives with efficient dataloader
        return zs.view(n_views, -1, zs.shape[-1])
    
    def get_std_cov_losses(self, z):
        """
        Calculate std and cov losses for a single view (of all examples).
        """
        z = z - z.mean(dim=0)
        std_z = torch.sqrt(z.var(dim=0).type(torch.float32) + 0.0001)   # avoid grad overflow with 32bit
        std_loss = torch.mean(torch.relu(1.0 - std_z), dtype=torch.float32)
        cov_z = (z.T @ z) / (z.size(0) - 1.0)
        cov_loss = off_diagonal(cov_z).pow_(2).sum().div(z.size(-1))
        return std_loss, cov_loss

    def loss(self, zs, lambd):
        # Get MSE between query and each key view (positives)
        z_q, z_ks = self.get_query_and_keys(zs)
        mse_losses = self.get_mse_losses(z_q, z_ks)

        # Gather across GPUS
        zs = torch.concat(zs, dim=0)        # concat to have a single gather operation
        if self.args.world_size > 1:
            zs = torch.cat(FullGatherLayer.apply(zs), dim=0)
        zs = self.split_z_into_views(zs)    # split back into views

        # Calculate std and cov losses
        std_loss = 0.0
        cov_loss = 0.0
        for z in zs:
            s_l, c_l = self.get_std_cov_losses(z)
            std_loss = std_loss + s_l
            cov_loss = cov_loss + c_l
            if self.args.add_cont_neg:
                s_l_c, c_l_c = self.get_std_cov_losses(z[:, self.z_split_inds[0]])
                std_loss = std_loss + s_l_c
                cov_loss = cov_loss + c_l_c
        std_loss = std_loss / len(zs)
        cov_loss = cov_loss / len(zs)

        # Final loss
        mse_loss = sum([lmd * mse_l for lmd, mse_l in zip(lambd, mse_losses)]) / len(mse_losses)
        entropy_term = self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss
        loss = mse_loss + entropy_term
        
        # Logs
        logs = dict(loss=loss.detach().clone(), entropy_term=entropy_term.detach(), 
                    std_loss=std_loss.detach(), cov_loss=cov_loss.detach())
        logs.update({f"mse_loss_{i}": mse_loss.detach() for i, mse_loss in enumerate(mse_losses)})
        logs.update({f"inv_constraint_{i}": pos_loss.detach() for i, pos_loss in enumerate(mse_losses)})

        return loss, logs


######################################################################################################
######################## BASELINE ALGORITHMS #########################################################
######################################################################################################

class SimCLRLooC(SimCLRMultiHead):
    """Leave-one-out method of Xiao et al., 2021 (https://arxiv.org/abs/2008.05659).
    
    Notes:
        - no code was provided in the paper, so this implementation is based on the method description.
        - not yet reproduced the results.
    """
    def __init__(self, args, backbone, projector):
        super().__init__(args, backbone, projector)
        self.precompute_mask_params()  # for selecting "extra negatives"

    def precompute_mask_params(self):
        self.total_images = self.args.batch_size * self.n_views
        self.batch_size_gpu = self.total_images // self.args.world_size
        self.orig_images_gpu = self.batch_size_gpu // self.n_views
    
    def get_mask_indices(self, view_i, view_j):
        """ Get a list of the pos_mask indices corresponding to the view-i vs. view-j comparison (for gpu_id=rank)."""
        view_inds = [view_i, view_j]
        orig_members = torch.arange(self.orig_images_gpu)
        
        row_inds, col_inds = [], []
        for anchor in view_inds:
            pos = view_inds[1 - view_inds.index(anchor)]    # other view
            col_inds.append(self.batch_size_gpu * self.args.rank + pos * self.orig_images_gpu + orig_members)
            row_inds.append(torch.arange(anchor * self.orig_images_gpu, (anchor + 1) * self.orig_images_gpu).long())
        
        row_inds = torch.vstack(row_inds).T.flatten()     # shape: (o_g*2,)
        col_inds = torch.vstack(col_inds).T.flatten()     # shape: (o_g*2,)
        
        return row_inds, col_inds

    def loss(self, zs, lambd):
        # Extract query and keys: single query, multiple keys
        z_q, z_ks = self.get_query_and_keys(zs)         # z_q: [batch_size_gpu, z_dim]
        
        # Logging only: get MSE (before normalization) for each space
        mse_losses = []
        with torch.no_grad():
            mse_losses = self.get_mse_losses(z_q, z_ks)

        # Concat, normalize and gather
        zs = torch.concat(zs, dim=0)        # [n_views, batch_size_gpu, z_dim] --> [n_views*batch_size_gpu, z_dim]
        zs = self.normalize_per_zspace(zs)  # normalize each space separately, joint norm = sqrt(n_spaces)
        if self.args.world_size > 1:
            zs_dist = torch.cat(FullGatherLayer.apply(zs), dim=0) # [n_views*batch_size_gpu*world_size, z_dim]
        else:
            zs_dist = zs

        # Per-space losses
        pos_losses = []     # store positive losses for each space before averaging (constraint analysis)
        neg_loss = 0.       # average negative loss over spaces (no analysis)
        for s in range(self.n_spaces):
            # Similarity matrix for space s: [batch_size_gpu, batch_size_gpu*world_size]
            zs_s = zs[:, self.z_split_inds[s]]
            zs_dist_s = zs_dist[:, self.z_split_inds[s]]
            sim_s = torch.exp(torch.mm(zs_s, zs_dist_s.t()) / self.args.temperature)
            
            # Loss for space s
            if s == 0:  # Content space
                # Positive loss: all key-views are positives, compare them all to each other
                pos_loss_s = -torch.mean(torch.log(torch.sum(sim_s * self.pos_mask, dim=1)))

                # Negative loss: negatives are other instances, as usual
                neg_loss_s = -torch.mean(torch.log(torch.sum(sim_s * self.neg_mask, dim=1)))
            else:       # Style space
                # Get indices for positive similarities: (view0/query, view-s) and (view-s, view0/query) only
                pos_inds_s = self.get_mask_indices(0, s)  # [0,s] = sim(view0/query, view_s); includes both [0,s] and [s,0].
                
                # Positive loss
                pos_loss_s = -torch.mean(torch.log(sim_s[pos_inds_s]))  # no sum needed as there is only one element per column

                # Negative loss: all other instances + "extra negatives"
                # Extra negatives: all similarities except (view0/query, view-s) and (view-s, view0/query)
                neg_mask_space_s = self.neg_mask + self.pos_mask
                neg_mask_space_s[pos_inds_s] = 0.
                neg_loss_s = -torch.mean(torch.log(torch.sum(sim_s * neg_mask_space_s, dim=1)))
            
            pos_losses.append(pos_loss_s)
            neg_loss = neg_loss + neg_loss_s
        
        # Final loss
        pos_loss = sum([lmd * pos_loss for lmd, pos_loss in zip(lambd, pos_losses)]) / self.n_spaces
        neg_loss = neg_loss / self.n_spaces
        loss = pos_loss + neg_loss
        
        # Logs
        logs = dict(loss=loss.detach().clone(), entropy_term=neg_loss.detach())
        logs.update({f"mse_loss_{i}": mse_loss.detach() for i, mse_loss in enumerate(mse_losses)})
        logs.update({f"inv_constraint_{i}": pos_loss.detach() for i, pos_loss in enumerate(pos_losses)})

        return loss, logs
    
    def inv_constraint_loss(self, inv_constraint_value, *args, **kwargs):
        """
        Standardize or normalise the invariant constraint term/loss (positive term for SimCLR).

        To do so, remove the effect of:
        - 1) summing over multiple key views *for the content space only*; and
        - 2) temperature.
        
        Also add 1 to make it a positive loss (min=0).
        """
        if kwargs["space"] == 0:
            # Content space: multiple key views used
            return (inv_constraint_value - math.log(self.n_views - 1)) * self.args.temperature + 1.
        else:
            # Style spaces: single key view used
            return inv_constraint_value * self.args.temperature + 1.


class SimCLRAugSelf(SimCLR):
    """AugSelf method of Lee et al., 2021 (https://arxiv.org/abs/2111.09613)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Not implemented yet")


class VICRegAugSelf(VICReg):
    """VICReg version of the AugSelf method of Lee et al., 2021 (https://arxiv.org/abs/2111.09613)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Not implemented yet")
