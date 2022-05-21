from abc import ABC, abstractmethod

from .builder import HOOKS


class TrainDataHook(ABC):
    @abstractmethod
    def __call__(self, data, *args, **kwargs):
        return data


class EvalDataHook(ABC):
    @abstractmethod
    def __call__(self, data, *args, **kwargs):
        return data


class TrainOutputsHook(ABC):
    @abstractmethod
    def __call__(self, outputs, *args, **kwargs):
        return outputs


class EvalOutputsHook(ABC):
    @abstractmethod
    def __call__(self, outputs, *args, **kwargs):
        return outputs


@HOOKS.register_module()
class ADNeRFTrainDataHook(TrainDataHook):
    def __call__(self, data, iter, config, *args, **kwargs):
        if iter >= config.train_no_smooth_iters:
            data["use_smooth"] = True
        else:
            data["use_smooth"] = False
        return data


@HOOKS.register_module()
class ADNeRFEvalDataHook(EvalDataHook):
    def __call__(self, data, config, *args, **kwargs):
        if config.eval_use_smooth:
            data["use_smooth"] = True
        else:
            data["use_smooth"] = False

        return data


@HOOKS.register_module()
class SDNeRFTrainDataHook(ADNeRFTrainDataHook):
    pass


@HOOKS.register_module()
class SDNeRFEvalDataHook(ADNeRFEvalDataHook):
    pass


@HOOKS.register_module()
class SDNeRFOutputsHook(TrainOutputsHook, EvalOutputsHook):
    def __call__(self, outputs, *args, **kwargs):
        # resample head mask and compute loss
        return outputs
