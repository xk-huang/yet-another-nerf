from yanerf.utils.config import Config

model_cfg = Config.fromfile("{{ fileDirname }}/models/zero_outputer.yml")._cfg_dict.to_dict()
ray_sampler_cfg = Config.fromfile("{{ fileDirname }}/ray_samplers/ray_sampler.yml")._cfg_dict.to_dict()
renderer_cfg = Config.fromfile(
    "{{ fileDirname }}/renderers/multipass_emission_absorption_renderer.yml"
)._cfg_dict.to_dict()

pipeline = dict(
    type="NeRFPipeline",
    **model_cfg,
    **ray_sampler_cfg,
    **renderer_cfg,
    chunk_size_grid=30,
    num_passes=2,
    loss_weights={
        "loss_rgb_mse": 1.0,
        "loss_prev_stage_rgb_mse": 1.0,
    },
    output_rasterized_mc=True,
)

del Config, model_cfg, ray_sampler_cfg, renderer_cfg
