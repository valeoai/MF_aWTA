# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5, temperature=1.0, wta_weights=None,dist_mode='ade', my_distance=None, velo_distance=None):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if True:
        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
            #print('I am here')
        else:
            #print("calculate nearest_mode_idxs based on predictions.")
            if my_distance is None:
                distance = (pred_trajs[:, :, :, 0:2].detach() - gt_trajs[:, None, :, :]).norm(dim=-1) #norm over (x,y) [batch, num_modes, timestamps]
                if dist_mode =='fde':
                    last_valid_idx = gt_valid_mask.sum(-1).clone() - 1
                    last_valid_idx = last_valid_idx.view(-1,1,1).repeat(1,distance.shape[1],1).long()
                    distance = torch.gather(distance, -1, last_valid_idx)[:,:,0] #take the last avaiable timestep
                    #print("fde mode: ", distance.shape)
                else:
                    distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) #sum over timestamps [batch, num_modes]
            else:
                #print('use last 2:', my_distance.sum())
                distance = my_distance.clone()
            nearest_mode_idxs = distance.argmin(dim=-1) #arg mean over timestamps [batch,]
        #assert False
        #print("nearest_mode_idxs.shape ", nearest_mode_idxs.shape)
        #print("pred_trajs.shape", pred_trajs.shape)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]
        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)
        gt_valid_mask_bis = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask_bis = gt_valid_mask_bis * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)
        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask_bis).sum(dim=-1)
        reg_loss_classic = reg_loss.detach().clone()
    if wta_weights is not None:
        del reg_loss
        #nearest_mode_idxs = pre_nearest_mode_idxs
        #assert pre_nearest_mode_idxs is not None
        #print("in AWTA")
        #nearest_mode_idxs = None
        if my_distance is None:
            distance = (pred_trajs[:, :, :, 0:2].detach() - gt_trajs[:, None, :, :]).norm(dim=-1) #norm over (x,y)
            if dist_mode == 'fde':
                last_valid_idx = gt_valid_mask.sum(-1) - 1
                last_valid_idx = last_valid_idx.view(-1,1,1).repeat(1,distance.shape[1],1).long()
                distance = torch.gather(distance, -1, last_valid_idx)[:,:,0] #take the last avaiable timestep
                #print(distance.shape)
            else:
                distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) #sum over timestamps [batch, num_modes]
        else:
            #print('use last 3:', my_distance.sum())
            distance = my_distance.clone()

        nearest_mode_idxs = distance.argmin(dim=-1) #arg mean over timestamps
        #print("nearest_mode_idxs.shape!!!! ", nearest_mode_idxs.shape)
        #nearest_mode_idxs = None
        assert (gt_valid_mask.sum(-1) >0.0).all()
        if dist_mode == 'ade':
            avg_distance = distance/gt_valid_mask.sum(-1)[:,None] # (batch, num_modes)
        else:
            #print("fde")
            avg_distance = distance
        if velo_distance is not None:
            #print("velo_distance", velo_distance)
            avg_distance += velo_distance
        #print("avg_distance[0]", avg_distance[0])
        #print(temperature)
        wta_weights = torch.softmax(-1.0*avg_distance/temperature, dim=-1).detach() #(batch, num_modes)

        #nearest_mode_bs_idxs = torch.arange(batch_size).long().to(pred_trajs.device)  # (batch_size, 2)?
        #print("nearest_mode_bs_idxs.shape ", nearest_mode_bs_idxs.shape)
    
        # dummy implementation with for loop
        reg_loss = 0
        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]
        for mode_index in range(wta_weights.shape[1]):
            wta_weight = wta_weights[:, mode_index]
            nearest_trajs = pred_trajs[:, mode_index]  # (batch_size, num_timestamps, 5)
            res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
            dx = res_trajs[:, :, 0]
            dy = res_trajs[:, :, 1]

            if use_square_gmm:
                log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
                std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
                rho = torch.zeros_like(log_std1)
            else:
                log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
                log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
                std1 = torch.exp(log_std1)  # (0.2m to 150m)
                std2 = torch.exp(log_std2)  # (0.2m to 150m)
                rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

            #gt_valid_mask = gt_valid_mask.type_as(pred_scores)
            #if timestamp_loss_weight is not None:
            #    gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

            # -log(a^-1 * e^b) = log(a) - b
            reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
            reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                    (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                    std1 * std2))  # (batch_size, num_timestamps)

            reg_loss += ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)*wta_weight #loss for this mode
       
    return reg_loss, reg_loss_classic, nearest_mode_idxs, wta_weights
