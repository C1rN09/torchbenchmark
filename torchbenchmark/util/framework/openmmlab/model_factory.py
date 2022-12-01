import os
import torch
from mmengine.hub import get_config, get_model
from mmengine.registry import DefaultScope
from mmengine.runner import Runner
from torchbenchmark import DATA_PATH
from torchbenchmark.util.model import BenchmarkModel
from typing import Tuple, Generator, Optional


class OpenMMLabModel(BenchmarkModel):
    # To recognize this is a openmmlab model
    OPENMMLAB_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default train/eval precision on CUDA device are fp32
    DEFAULT_TRAIN_CUDA_PRECISION = 'fp32'
    DEFAULT_EVAL_CUDA_PRECISION = 'fp32'

    def __init__(self, cfg_path, test, device, jit=False,
                 batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit,
            batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        scope = cfg_path.split(':')[0]
        config = get_config(cfg_path=cfg_path, pretrained=True)
        self._modify_data_root(config)
        self.model = get_model(cfg_path=cfg_path, pretrained=True).to(device)
        if test == 'train':
            data_loader = config.train_dataloader
            self.model.train()
        elif test == 'eval':
            data_loader = config.val_dataloader
            self.model.eval()
        data_loader.batch_size = self.batch_size
        with DefaultScope.overwrite_default_scope(scope):
            self.data_loader = Runner.build_dataloader(data_loader)
        self.example_inputs = iter(self.data_loader).__next__()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            capturable=bool(int(os.getenv("ADAM_CAPTURABLE", 0)))
        )
    
    @staticmethod
    def _modify_data_root(config):
        def _modify(dataset):
            mapping = {
                'coco': 'mm_coco_mini',
                'imagenet': 'mm_imagenet_mini'
            }
            name = dataset.data_root.strip('/').split('/')[-1]
            name = mapping[name]
            dataset.data_root = os.path.join(DATA_PATH, name)
        _modify(config.train_dataloader.dataset)
        _modify(config.val_dataloader.dataset)

    def gen_inputs(self, num_batches:int=1) -> Tuple[Generator, Optional[int]]:
        def _gen_inputs():
            while True:
                result = []
                for _i in range(num_batches):
                    result.append((torch.randn((self.batch_size, 3, 224, 224)).to(self.device),))
                if self.dargs.precision == "fp16":
                    result = list(map(lambda x: (x[0].half(), ), result))
                yield result
        return (_gen_inputs(), None)

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        self.optimizer.zero_grad()
        data = self.example_inputs
        data = self.model.data_preprocessor(data, True)
        losses = self.model._run_forward(data, mode='loss')
        loss, _ = self.model.parse_losses(losses)
        loss.backward()
        self.optimizer.step()
    
    def eval(self):
        data = self.example_inputs
        data = self.model.data_preprocessor(data, True)
        out = self.model._run_forward(data, mode='predict')
        return tuple(out)