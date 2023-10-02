import torch
from utils import *
from trainer import Base_Trainer


class Phase1_Trainer(Base_Trainer):
    def __init__(self, model, criterion, optimizer, config,
                 train_loader, valid_loader, scheduler, run):
        super().__init__(run, config)

        self.mode = config['mode']
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.class_num = 16 if 'concept' in self.mode else 20

    def train_epoch(self):
        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
            if self.use_train:
                self.model.train()
                train_total_loss = []
                train_count = 0
                train_total_acc = 0
                for input, output, task in self.train_loader:
                    train_batch_size = input.shape[0]
                    train_count += train_batch_size
                    input = input.to(torch.float32).to('cuda')
                    output = output.to(torch.float32).to('cuda')
                    task = task.to(torch.long).to('cuda')

                    output = self.model(input, output)

                    if 'multi-bc' in self.mode:
                        for i in range(task.shape[1]):
                            loss = self.criterion(nn.functional.sigmoid(output), task.permute(1,0,2).to(torch.float32))
                    else:
                        loss = self.criterion(output, task)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_total_loss.append(loss)

                    if 'multi-bc' in self.mode:
                        temp_list = emr(train_batch_size, self.class_num, output, task)
                        train_total_acc += torch.tensor(temp_list).sum()
                    else:
                        output_softmax = torch.softmax(output, dim=1)
                        output_argmax = torch.argmax(output_softmax, dim=1)
                        train_total_acc += output_argmax.eq(task).sum().item()
                self.scheduler.step()

                if self.use_wandb:
                    wandb.log({
                        "train_loss": sum(train_total_loss) / len(train_total_loss),
                        "train_acc": train_total_acc / train_count,
                        "train_correct_num": train_total_acc,
                    }, step=epoch)
                print(f'train loss: {sum(train_total_loss) / len(train_total_loss)}')
                print(f'train acc: {train_total_acc / train_count}({train_total_acc}/{train_count})')

                if not self.use_pretrain and epoch % 50 == 0:
                    self.save_checkpoint(self.model, f'Phase1_{self.model_name}_temp.pt')

            if self.use_valid:
                self.valid_epoch(pbar, epoch)


        if not self.use_pretrain:
            file_name = f'Phase1_{self.model_name}_b{self.train_batch_size}_lr{self.lr}_{self.epochs}.pt'
            self.save_checkpoint(self.model, file_name)

    def valid_epoch(self, pbar, epoch):
        valid_count = 0
        valid_total_acc = 0
        valid_total_loss = []
        self.model.eval()
        for input, output, task in self.valid_loader:
            valid_batch_size = input.shape[0]
            valid_count += input.shape[0]
            input = input.to(torch.float32).to('cuda')
            output = output.to(torch.float32).to('cuda')
            task = task.to(torch.long).to('cuda')

            output = self.model(input, output)

            if 'multi-bc' in self.mode:
                loss = self.criterion(nn.functional.sigmoid(output), task.permute(1,0,2).to(torch.float32))
            else:
                loss = self.criterion(output, task)
            valid_total_loss.append(loss)

            if 'multi-bc' in self.mode:
                temp_list = emr(valid_batch_size, self.class_num, output, task)
                valid_total_acc += torch.tensor(temp_list).sum()
            else:
                output_softmax = torch.softmax(output, dim=1)
                output_argmax = torch.argmax(output_softmax, dim=1)
                valid_total_acc += output_argmax.eq(task).sum().item()

        print(f'valid loss: {sum(valid_total_loss) / len(valid_total_loss)}')
        print(f'valid acc: {valid_total_acc / valid_count}({valid_total_acc}/{valid_count})')
        if self.use_wandb:
            wandb.log({
                "valid_loss": sum(valid_total_loss) / len(valid_total_loss),
                "valid_acc": valid_total_acc / valid_count,
                "valid_correct_num": valid_total_acc,
            }, step=epoch)
