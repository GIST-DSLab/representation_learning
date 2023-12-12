import torch
import torch.nn as nn

class custom_embedding_layer(nn.Module):
    def __init__(self, padding_class, num_class, embedded_dim):
        super().__init__()
        self.padding_class = padding_class
        self.embedding_parameters = nn.Embedding(num_class, embedded_dim, padding_idx=0)
        self.zero_weight = torch.zeros(1, embedded_dim).to('cuda')

    def forward(self, x):
        x_embedding = self.embedding_parameters(x)
        filter_padding = torch.where(x == self.padding_class, True, False)
        x_embedding[filter_padding] = self.zero_weight

        return x_embedding

class vae_v1_1(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.embedding_size = 3
        self.first_size = 2
        self.latent_size = 1
        self.embedding = nn.Embedding(self.num_class, self.embedding_size, padding_idx=self.padding_number)

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.first_size),
            nn.ReLU(),
            nn.Linear(self.first_size, self.latent_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.first_size),
            nn.ReLU(),
            nn.Linear(self.first_size, self.embedding_size),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(self.latent_size, self.latent_size)
        self.sigma_layer = nn.Linear(self.latent_size, self.latent_size)

        self.proj = nn.Linear(self.embedding_size, self.num_class)

    def forward(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11).permute(0,3,1,2)

        return output

    def for_preprocessing(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps

        return latent_vector

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 1, 1)

# 실패
class compressed_cnn_vae(nn.Module): # embedding size 관련 실험
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.embedding_size = 3
        self.h_size = 256
        self.z_size = 128
        self.embedding = nn.Embedding(self.num_class, self.embedding_size, padding_idx=self.padding_number)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.embedding_size, 4, 4, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 4, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=0, stride=1),
            nn.ReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(self.z_size, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, self.embedding_size, kernel_size=6, stride=2),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(self.h_size, self.z_size)
        self.sigma_layer = nn.Linear(self.h_size, self.z_size)
        # self.latent_layer = nn.Linear(self.z_size, self.h_size)

        self.proj = nn.Linear(self.embedding_size, self.num_class)

    def forward(self, x):
        embed_x = self.embedding(x.to(torch.long)).squeeze(1)
        feature_map = self.encoder(embed_x.permute(0,3,1,2).contiguous())
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        # latent_vector = self.latent_layer(latent_vector)
        output = self.decoder(latent_vector)
        output = self.proj(output.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        return output

    def for_preprocessing(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps

        return latent_vector 

class compressed_vae(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.embedding_size = 3
        self.h_size = 32
        self.z_size = 16
        self.embedding = nn.Embedding(self.num_class, self.embedding_size, padding_idx=self.padding_number)

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.h_size),
            nn.ReLU(),
            nn.Linear(self.h_size, self.z_size ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size , self.h_size),
            nn.ReLU(),
            nn.Linear(self.h_size, self.embedding_size),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(self.z_size, self.z_size)
        self.sigma_layer = nn.Linear(self.z_size, self.z_size)

        self.proj = nn.Linear(self.embedding_size, self.num_class )

    def forward(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps
        output = self.decoder(latent_vector)
        output = self.proj(output).reshape(-1,30,30,11).permute(0,3,1,2)

        return output

    def for_preprocessing(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            embed_x = self.embedding(x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.embedding(x.reshape(1, 900).to(torch.long))
        feature_map = self.encoder(embed_x)
        mu = self.mu_layer(feature_map)
        sigma = self.mu_layer(feature_map)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_vector = mu + std * eps

        return latent_vector


class vae_v1_2(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.embedding = custom_embedding_layer(self.padding_number, self.num_class, 512)

        self.encoder = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 256, bias=False),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128, bias=False)
        self.sigma_layer = nn.Linear(128, 128, bias=False)

    def forward(self, concat_input):
        concat_feature = self.encoder(concat_input)

        concat_mu = self.mu_layer(concat_feature)
        concat_sigma = self.mu_layer(concat_feature)
        concat_std = torch.exp(0.5 * concat_sigma)
        concat_eps = torch.randn_like(concat_std)
        concat_latent_vector = concat_mu + concat_std * concat_eps

        output = self.decoder(concat_latent_vector)


        return output

class vae_v1(nn.Module):
    def __init__(self, model_file):
        super().__init__()
        self.vae_v1_1 = vae_v1_1(model_file)
        self.vae_v1_2 = vae_v1_2(model_file)

    def forward(self, x):
        if len(x.shape) >= 3:
            batch_size = x.shape[0]
            # embed_x = self.vae_v1_1.embedding(x.reshape(batch_size, 900).to(torch.long))
            embed_x = self.vae_v1_1.embedding(x.to(torch.long))
        else:
            embed_x = self.vae_v1_1.embedding(x.reshape(1, 900).to(torch.long))
        e1_feature_map = self.vae_v1_1.encoder(embed_x)
        e1_mu = self.vae_v1_1.mu_layer(e1_feature_map)
        e1_sigma = self.vae_v1_1.mu_layer(e1_feature_map)
        e1_std = torch.exp(0.5 * e1_sigma)
        e1_eps = torch.randn_like(e1_std)
        e1_latent_vector = e1_mu + e1_std * e1_eps

        print(1)


class vae_v2(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.num_example = 10
        self.embedding = custom_embedding_layer(self.padding_number, self.num_class, 512)

        self.encoder1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.mu_layer1 = nn.Linear(128, 128)
        self.sigma_layer1 = nn.Linear(128, 128)

        self.mu_layer2 = nn.Linear(32, 32)
        self.sigma_layer2 = nn.Linear(32, 32)

        self.proj = nn.Linear(512, 11)

    def forward(self, input, output):
        batch_size = input.shape[0]
        input_embedding = self.embedding(input.reshape(batch_size, -1).to(torch.long))
        output_embedding = self.embedding(output.reshape(batch_size, -1).to(torch.long))

        input_feature = self.encoder1(input_embedding)
        output_feature = self.encoder1(output_embedding)

        input_mu = self.mu_layer1(input_feature)
        input_sigma = self.mu_layer1(input_feature)
        input_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(input_std)
        input_latent_vector = input_mu + input_std * input_eps

        output_mu = self.mu_layer1(output_feature)
        output_sigma = self.mu_layer1(output_feature)
        output_std = torch.exp(0.5 * output_sigma)
        output_eps  = torch.randn_like(output_std)
        output_latent_vector = output_mu + output_std * output_eps

        concat_value = torch.cat((input_latent_vector, output_latent_vector),dim=1)

        concat_feature = self.encoder2(concat_value)

        concat_mu = self.mu_layer2(concat_feature)
        concat_sigma = self.mu_layer2(concat_feature)
        concat_std = torch.exp(0.5 * concat_sigma)
        concat_eps = torch.randn_like(concat_std)
        concat_latent_vector = concat_mu + concat_std * concat_eps

        concat_output = self.decoder2(concat_latent_vector)

        input_output = self.decoder1(concat_output[:,:9000,:]) # concat_output을 index slicing해서 줘야함.
        output_output = self.decoder1(concat_output[:,9000:,:])

        input_pred = self.proj(input_output).reshape(batch_size, self.num_example, 30,30,11)
        output_pred = self.proj(output_output).reshape(batch_size, self.num_example,30,30,11)

        return input_pred, output_pred


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        self.color_convert = {
            0: [0,0,0],         # black

            1: [0, 116, 217],   # blue
            2: [255, 65, 54],   # red
            3: [46, 204, 64],   # green
            4: [255, 220, 0],   # yellow
            5: [170, 170, 170], # grey
            6: [240, 18, 190],  # fuschia
            7: [255, 133, 27],  # orange
            8: [127, 219, 255], # teal
            9: [135, 12, 37]    # brown
        }

    def preprocessing(self):
        pass

    def forward(self):
        pass