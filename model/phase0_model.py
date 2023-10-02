import torch
import torch.nn as nn


class custom_embedding_layer(nn.Module):
    def __init__(self, padding_class, num_class, embedded_dim):
        super().__init__()
        self.padding_class = padding_class
        self.embedding_parameters = nn.Embedding(num_class, embedded_dim)
        self.zero_weight = torch.zeros(1, embedded_dim).to('cuda')

    def forward(self, x):
        x_embedding = self.embedding_parameters(x)
        filter_padding = torch.where(x == self.padding_class, True, False)
        x_embedding[filter_padding] = self.zero_weight

        return x_embedding

class vae_v1(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.padding_number = 0
        self.num_class = 11
        self.embedding = custom_embedding_layer(self.padding_number, self.num_class, 512)

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128)
        self.sigma_layer = nn.Linear(128, 128)

        self.proj = nn.Linear(512, 11)

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



