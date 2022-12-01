from torchbenchmark.util.framework.openmmlab.model_factory import OpenMMLabModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(OpenMMLabModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 256
    DEFAULT_EVAL_BSIZE = 256

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(
            cfg_path='mmcls::deit/deit-small_pt-4xb256_in1k.py', test=test,
            device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
