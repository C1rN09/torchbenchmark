import os
import torch
from mmengine.hub import get_config, get_model
from mmengine.registry import DefaultScope
from mmengine.runner import Runner
from torchbenchmark import DATA_PATH
from torchbenchmark.util.model import BenchmarkModel
import types
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
            self.forward_mode = 'loss'
        elif test == 'eval':
            data_loader = config.val_dataloader
            self.model.eval()
            self.forward_mode = 'predict'
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

    def get_module(self):
        # Currently some OpenMMLab `data_preprocessor` failed dynamo, so we
        # apply it before forward
        # TODO(C1rN09): remove this logic if data_preprocessor is supported
        example_inputs = self.model.data_preprocessor(self.example_inputs, True)
        example_inputs['mode'] = self.forward_mode
        # Replace forward function to be compatible with dynamo benchmark
        self.model._old_forward = self.model.forward
        def _new_forward(self, *args, **kwargs):
            assert len(args) == 1 and isinstance(args[0], dict)
            data = args[0]
            return self._old_forward(**data, **kwargs)
        self.model.forward = types.MethodType(_new_forward, self.model)
        return self.model, (example_inputs,)

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