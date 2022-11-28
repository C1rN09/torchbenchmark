import os
import torch
from mmengine.hub import get_config, get_model
from mmengine.registry import DefaultScope
from mmengine.runner import Runner
from torchbenchmark import DATA_PATH
from torchbenchmark.util.model import BenchmarkModel


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
        scope = cfg_path.split(':')[0]
        config = get_config(cfg_path=cfg_path, pretrained=True)
        if test == 'train':
            batch_size = config.train_dataloader.batch_size
        elif test == 'eval':
            batch_size = config.val_dataloader.batch_size

        super().__init__(test=test, device=device, jit=jit,
            batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.model = get_model(cfg_path=cfg_path, pretrained=True).to(device)
        self._modify_data_root(config)
        with DefaultScope.overwrite_default_scope(scope):
            self.train_loader = Runner.build_dataloader(config.train_dataloader)
            self.val_loader = Runner.build_dataloader(config.val_dataloader)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            capturable=bool(int(os.getenv("ADAM_CAPTURABLE", 0)))
        )
        if test == 'train':
            self.model.train()
            self.example_inputs = iter(self.train_loader).__next__()
        elif test == 'eval':
            self.model.eval()
            self.example_inputs = iter(self.val_loader).__next__()
    
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

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        data = self.example_inputs
        data = self.model.data_preprocessor(data, True)
        losses = self.model._run_forward(data, mode='loss')
        loss, _ = self.model.parse_losses(losses)
        loss.backward()
        self.optimizer.step()
    
    def eval(self):
        with torch.no_grad():
            data = self.example_inputs
            data = self.model.data_preprocessor(data, True)
            out = self.model._run_forward(data, mode='predict')
        return tuple(out)