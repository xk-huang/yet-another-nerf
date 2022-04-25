import torch

from yanerf.pipelines.ray_samplers import RAY_SAMPLERS
from yanerf.pipelines.ray_samplers.utils import EvaluationMode
from yanerf.utils.config import Config


def test_on_cuda():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    test_ray_sampler()


def test_ray_sampler():
    ray_sampler_cfg = Config.fromfile("tests/configs/pipelines/ray_samplers/ray_sampler.yml")
    ray_sampler = RAY_SAMPLERS.build(ray_sampler_cfg.ray_sampler)

    batch_size = 2
    poses = torch.randn(batch_size, 3, 4)
    focal_lengths = torch.ones(batch_size) * 500
    n_rays_per_image_sampled_from_mask = ray_sampler_cfg.ray_sampler.n_rays_per_image_sampled_from_mask
    n_pts_per_ray_training = ray_sampler_cfg.ray_sampler.n_pts_per_ray_training
    n_pts_per_ray_evaluation = ray_sampler_cfg.ray_sampler.n_pts_per_ray_training
    image_width, image_height = ray_sampler_cfg.ray_sampler.image_width, ray_sampler_cfg.ray_sampler.image_height
    min_depth, max_depth = ray_sampler_cfg.ray_sampler.min_depth, ray_sampler_cfg.ray_sampler.max_depth
    img = torch.randn(batch_size, 3, image_height, image_width)

    gt_shapes = {
        EvaluationMode.TRAINING: {
            "origins": [batch_size, n_rays_per_image_sampled_from_mask, 1, 3],
            "directions": [batch_size, n_rays_per_image_sampled_from_mask, 1, 3],
            "lengths": [batch_size, n_rays_per_image_sampled_from_mask, 1, n_pts_per_ray_training],
            "xys": [batch_size, n_rays_per_image_sampled_from_mask, 1, 2],
        },
        EvaluationMode.EVALUATION: {
            "origins": [batch_size, image_height, image_width, 3],
            "directions": [batch_size, image_height, image_width, 3],
            "lengths": [batch_size, image_height, image_width, n_pts_per_ray_evaluation],
            "xys": [batch_size, image_height, image_width, 2],
        },
    }

    outs = ray_sampler(poses, focal_lengths, EvaluationMode.TRAINING)
    for k, v in outs._asdict().items():
        print(f"{k}: {v.shape}")
        assert list(v.shape) == gt_shapes[EvaluationMode.TRAINING][k]
    assert outs._asdict()["lengths"].min() >= min_depth and outs._asdict()["lengths"].max() <= max_depth

    _min_depth, _max_depth = 15, 30
    outs = ray_sampler(poses, focal_lengths, EvaluationMode.TRAINING, min_depth=_min_depth, max_depth=_max_depth)
    for k, v in outs._asdict().items():
        print(f"{k}: {v.shape}")
        assert list(v.shape) == gt_shapes[EvaluationMode.TRAINING][k]
    assert outs._asdict()["lengths"].min() >= _min_depth and outs._asdict()["lengths"].max() <= _max_depth

    _image_height, _image_width = 3, 6
    outs = ray_sampler(
        poses, focal_lengths, EvaluationMode.EVALUATION, image_height=_image_height, image_width=_image_width
    )
    for k, v in outs._asdict().items():
        print(f"{k}: {v.shape}")
        assert list(v.shape[:-1]) == [batch_size, _image_height, _image_width]

    outs = ray_sampler(
        poses,
        focal_lengths,
        EvaluationMode.EVALUATION,
        min_depth=_min_depth,
        max_depth=_max_depth,
        image_height=_image_height,
        image_width=_image_width,
    )
    for k, v in outs._asdict().items():
        print(f"{k}: {v.shape}")
        assert list(v.shape[:-1]) == [batch_size, _image_height, _image_width]
    assert outs._asdict()["lengths"].min() >= _min_depth and outs._asdict()["lengths"].max() <= _max_depth

    outs = ray_sampler(poses, focal_lengths, EvaluationMode.EVALUATION)
    for k, v in outs._asdict().items():
        print(f"{k}: {v.shape}")
        assert list(v.shape) == gt_shapes[EvaluationMode.EVALUATION][k]

        # assert list(v.shape) == gt_shapes[EvaluationMode.EVALUATION][k]

    # About sample gt image grid
    xys = outs[-1]
    xys_normalized = (xys / torch.Tensor([image_width - 1.0, image_height - 1.0])[None, None, None]) * 2 - 1
    sampled_img = torch.nn.functional.grid_sample(img, xys_normalized, align_corners=True)
    print(f"Error: align_corners=True {(sampled_img - img).abs().mean()}")
    sampled_img = torch.nn.functional.grid_sample(img, xys_normalized, align_corners=False)
    print(f"Error: align_corners=False {(sampled_img - img).abs().mean()}")

    spatial_shape = img.shape[2:]
    flat_img = img.view(batch_size, 3, -1)
    flat_xys = xys.view(batch_size, -1, 2)
    flat_xys = flat_xys[:, :, 0] + flat_xys[:, :, 1] * image_width
    flat_xys = flat_xys[:, None, :].expand(-1, 3, -1)
    gathered_img = torch.gather(flat_img, -1, flat_xys.long())
    gathered_img = gathered_img.view(batch_size, 3, *spatial_shape)
    assert torch.allclose(img, gathered_img)
    print(f"Error: gather tensor {(gathered_img - img).abs().mean()}")
