import torch
from abc import abstractmethod

class Base_Trainer:
    """
    Base class for all trainers
    """
    def __init__(self, run, config):
        self.epochs = config['epochs']
        self.checkpoint_dir = config['save_dir']
        self.run = run

        self.pretrain_file = config['pretrain_file']
        self.epochs = config['epochs']
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['valid_batch_size']
        self.lr = config['lr']

        self.use_permute = config['use_permute']
        self.use_pretrain = config['use_pretrain']
        self.use_train = config['use_train']
        self.use_valid = config['use_valid']
        self.use_batch = config['use_batch']
        self.use_size = config['use_size']
        self.use_wandb = config['use_wandb']

        self.model_name = config['model_name']
        self.mode_name = config['mode']
        self.device = f'cuda:{config["cuda_num"]}' if type(config['cuda_num']) == int else 'cpu'

        self.total_acc = 0
        self.count = 0


    @abstractmethod
    def train_epoch(self):
        """
        Training logic for an epoch

        train 수행할 loop, overwrite하기
        """
        raise NotImplementedError

    def save_checkpoint(self, model, file_name):
        """
        Saving checkpoints

        모델 저장
        """
        torch.save(model.state_dict(), f'{self.checkpoint_dir}/{file_name}')
