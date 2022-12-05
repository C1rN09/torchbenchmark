from torchbenchmark.util.framework.openmmlab.model_factory import OpenMMLabModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(OpenMMLabModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(
            cfg_path='mmcls::mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py', test=test,
            device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
