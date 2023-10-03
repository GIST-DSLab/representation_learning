from model import *
from dataset import *
from torch.utils.data import DataLoader
from trainer import *
from utils import *
import argparse
import yaml

def main(config, args):
    # get config infomation
    # config 정보 불러오기
    seed = config['seed']

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    model_file = config['model_file']

    lr = float(config['lr'])
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    valid_batch_size = config['valid_batch_size']
    kind_of_loss = config['kind_of_loss'].lower()
    optimizer_name = config['optimizer'].lower()
    scheduler_name = config['scheduler'].lower()
    lr_lambda = config['lr_lambda']
    step_size = config['step_size']
    gamma = config['gamma']

    use_permute = config['use_permute']
    use_rotate = config['use_rotate']
    use_wandb = config['use_wandb']
    use_pretrain = config['use_pretrain']

    trainer_name = config['trainer']
    train_dataset_name = config['train_data']
    valid_dataset_name = config['valid_data']

    mode = config['mode']
    device = f'cuda:{config["cuda_num"]}' if type(config['cuda_num']) == int else 'cpu'

    # setup data_loader instances
    # dataloader 설정
    train_dataset = globals()[dataset_name](train_dataset_name, mode=mode, permute_mode=use_permute, rotate_mode=use_rotate)
    valid_dataset = globals()[dataset_name](valid_dataset_name, mode=mode)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, shuffle=True)

    # setup wandb
    # wandb 설정
    if use_wandb:
        run = set_wandb(seed, mode, kind_of_loss, lr, epochs, train_batch_size, valid_batch_size, model_name, use_permute,
                        optimizer_name, use_pretrain)
    else:
        run = None

    # setup model
    # 모델 설정
    model = globals()[model_name](model_file).to(device)
    if use_pretrain:
        pretrain_file = config['pretrain_file']
        model.load_state_dict(torch.load(pretrain_file))

    # setup function handles of loss and metrics
    # loss함수와 metrics 설정
    criterion = set_loss(kind_of_loss).to(device)

    # setup optimizer and learning scheduler
    # optimizer와 learning scheduler 설정
    optimizer = set_optimizer(optimizer_name, model, lr)
    scheduler = set_lr_scheduler(optimizer, scheduler_name, lr_lambda, step_size, gamma)

    trainer = globals()[trainer_name](model, criterion, optimizer, config,
                      train_loader, valid_loader, scheduler, run)

    trainer.train_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed_fix(config['seed'])
    main(config, args)