from torchbenchmark.util.framework.openmmlab.model_factory import OpenMMLabModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(OpenMMLabModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(
            cfg_path='mmdet::faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py', test=test,
            device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
