
def body_height_termination(env, asset_cfg, height_threshold):
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    heights = asset.data.body_pos_w[:, body_ids, 2]
    fallen = torch.any(heights < height_threshold, dim=1)
    return fallen
