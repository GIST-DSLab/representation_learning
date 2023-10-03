import torch
from utils import *
from trainer import Base_Trainer


class Phase2_Trainer(Base_Trainer):
    def __init__(self, model, criterion, optimizer, config,
                 train_loader, valid_loader, scheduler, run):
        super().__init__(run, config)

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer

        self.LFLAG = True #TODO LFLAG - Linear로 구성된 autoencoder 사용시 True 아니면 False

    def train_epoch(self):
        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
            if self.use_train:
                train_loss = []
                for data, size in self.train_loader:
                    total_loss = []
                    data = torch.tensor(data).to(torch.float32).to(self.device)
                    if self.use_batch:
                        data = data.unsqueeze(1)

                    output = self.model(data)
                    if not self.LFLAG:
                        output = output.unsqueeze(-1)
                        # output = self.model.proj(output)
                        if data.shape[0] > 1:
                            output = output.permute(0,4,2,3,1)#.squeeze(4)
                        else:
                            output = output.permute(3, 1, 2, 0)
                        output = self.model.proj(output)
                        if data.shape[0] > 1:
                            output = output.squeeze()

                    # output = output.view(self.batch_size, -1, 11)
                    # output = output.permute(0, 2, 1)
                    view_output = torch.argmax(torch.softmax(output, dim=1), dim=1)

                    data_grid = data[:, :size[0][0], :size[0][1]].detach().cpu() - 1
                    output_grid = torch.round(output[:, :size[0][0], :size[0][1]].detach().cpu() - 1)

                    # TODO self.use_batch일때 x, y size를 활용시 각 size가 다르더라도 loss를 계산할 수 있겠끔 구현하기 - for문 사용하면 될것 같음
                    if self.use_batch or not self.use_size:
                        if not self.use_size:
                            output = output.view(data.shape[0], 11, -1)
                            loss = self.criterion(output, data.view(data.shape[0], -1).to(torch.long))
                        else:
                            for i in range(data.shape[0]):
                                total_loss.append(self.criterion(output[i, :, :size[i][0], :size[i][1]].reshape(-1, size[i][0]*size[i][1]).T.squeeze(),data[i, :, :size[i][0], :size[i][1]].reshape(-1, size[i][0]*size[i][1]).T.squeeze().to(torch.long)))
                            loss = torch.stack(total_loss).mean()
                    else:
                        if not self.LFLAG:
                            output = output.permute(3,0,1,2) #TODO 지우기
                            output = output.squeeze()[:, :size[0][0], :size[0][1]].reshape(data.shape[0], 11, -1)
                            loss = self.criterion(output[:, :, :size[0][0] * size[0][1]],data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))
                        else:
                            loss = self.criterion(output[:, :, :size[0][0], :size[0][1]].reshape(1, 11, -1), data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))

                    # plt.imshow(y_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)
                    # plt.imshow(output_grid, cmap=grid_visual.cmap, norm=grid_visual.norm)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_loss.append(loss)
                self.scheduler.step()

                if self.use_wandb:
                    wandb.log({
                        "train_loss": sum(train_loss) / len(train_loss),
                    }, step=epoch)
                pbar.write(f'train_loss: {sum(train_loss) / len(train_loss)}')

                if not self.use_pretrain and epoch % 50 == 0:
                    self.save_checkpoint(self.model, f'Phase2_{self.model_name}_temp.pt')

            if self.use_valid:
                self.valid_epoch(pbar, epoch)


        if not self.use_pretrain:
            if not self.use_size:
                file_name = f'Phase2_No_size_{self.model_name}_b{self.batch_size}_lr{self.lr}_{self.epochs}.pt'
                self.save_checkpoint(self.model, file_name)
            else:
                file_name = f'Phase2_{self.model_name}_b{self.batch_size}_lr{self.lr}_{self.epochs}.pt'
                self.save_checkpoint(self.model, file_name)

    def valid_epoch(self, pbar, epoch):
        self.total_acc = 0
        self.count = 0
        valid_loss = []
        total_loss = []

        for data, size in self.valid_loader:
            data = torch.tensor(data).to(torch.float32).to(self.device)
            # if use_permute:
            #     for i in range(30):
            #         for j in range(30):
            #             data[:, i, j] = permute_color[data[:, i, j]]
            # data = recursive_exchange(data, permute_color, 0, [])
            # data = torch.where(data == -1, 0, data)

            output = self.model(data)
            if not self.LFLAG:
                output = output.unsqueeze(-1)
                # output = self.model.proj(output)

                if data.shape[0] > 1:
                    output = output.permute(0, 4, 2, 3, 1).squeeze(4)
                else:
                    output = output.permute(3,1,2,0)
                output = self.model.proj(output)  #TODO 지우기

            data_grid = data[:, :size[0][0], :size[0][1]].squeeze().detach().cpu() - 1
            output_grid = torch.round(output[:, :size[0][0], :size[0][1]]).squeeze().detach().cpu() - 1

            # TODO self.use_batch일때 x, y size를 활용시 각 size가 다르더라도 loss를 계산할 수 있겠끔 구현하기 - for문 사용하면 될것 같음
            if (data.shape[0] > 1) or not self.use_size:
                if not self.use_size:
                    output = output.view(data.shape[0], -1, 11)
                    loss = self.criterion(output, data.view(data.shape[0], -1).to(torch.long))
                else:
                    for i in range(data.shape[0]):
                        total_loss.append(self.criterion(
                            output[i, :, :size[i][0], :size[i][1]].reshape(-1, size[i][0] * size[i][1]).T.squeeze(),
                            data[i, :, :size[i][0], :size[i][1]].reshape(-1, size[i][0] * size[i][1]).T.squeeze().to(
                                torch.long)))
                    loss = torch.stack(total_loss).mean()
            else:
                if not self.LFLAG:
                    output = output.permute(3, 0, 1, 2) #TODO 지우기
                    loss = self.criterion(output.squeeze()[:, :size[0][0], :size[0][1]].reshape(-1, size[0][0] * size[0][1]).T.squeeze(),data[:, :size[0][0], :size[0][1]].reshape(1, -1).squeeze().to(torch.long))
                else:
                    loss = self.criterion(output[:, :, :size[0][0], :size[0][1]].reshape(1, 11, -1), data[:, :size[0][0], :size[0][1]].reshape(1, -1).to(torch.long))
            if (data.shape[0] > 1):
                output = torch.argmax(torch.softmax(output, dim=1), dim=1).reshape(data.shape[0], data.shape[1], data.shape[2])
            else:
                if self.LFLAG:
                    output = torch.argmax(torch.softmax(output, dim=1), dim=1).reshape(data.shape[0], data.shape[1], data.shape[2])
                else:
                    output = torch.argmax(torch.softmax(output, dim=0), dim=0).reshape(data.shape[0], data.shape[1], data.shape[2])

            view_output = output[:, :size[0][0], :size[0][1]] - 1

            valid_loss.append(loss)
            if torch.equal(view_output, data[:, :size[0][0], :size[0][1]] - 1):
                self.total_acc += 1
            else:
                pass
            # total_loss.append(loss)
            self.count += 1
        if self.use_wandb:
            wandb.log({
                "valid_loss": sum(valid_loss) / len(valid_loss),
                "valid_accuracy:": self.total_acc / self.count,
            }, step=epoch)

        pbar.write(f'acc: {self.total_acc / self.count}({self.total_acc}/{self.count})')
        print(sum(valid_loss) / len(valid_loss))
