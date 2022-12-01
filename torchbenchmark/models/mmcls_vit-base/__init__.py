from torchbenchmark.util.framework.openmmlab.model_factory import OpenMMLabModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(OpenMMLabModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(
            cfg_path='mmcls::vision_transformer/vit-base-p16_pt-32xb128-mae_in1k-224.py', test=test,
            device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
