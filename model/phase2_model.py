import torch
import torch.nn as nn
from .phase1_model import *

class vae_cl_solver(nn.Module):
    def __init__(self, vae_model_file, model_file):
        super().__init__()

        self.vae_cl = vae_classifier_v1(vae_model_file)
        self.vae_cl.load_state_dict(torch.load(model_file))
        self.vae_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128
        self.third_layer_parameter_size = 64
        self.last_parameter_size = 128
        self.num_categories = 16

        self.fusion_layer1 = nn.Linear(128*2, 128)

        self.solver_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(128, 128)
        self.sigma_layer = nn.Linear(128, 128)
        self.solver_proj = nn.Linear(512, 11)

    def vae_freeze(self):
        for param in self.vae_cl.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]
        if batch_size > 1:
            embed_x = self.vae_cl.autoencoder.embedding(input_x.reshape(batch_size, 900).to(torch.long))
        else:
            embed_x = self.vae_cl.autoencoder.embedding(input_x.reshape(1, 900).to(torch.long))

        input_feature = self.vae_cl.autoencoder.encoder(embed_x)
        input_mu = self.vae_cl.autoencoder.mu_layer(input_feature)
        input_sigma = self.vae_cl.autoencoder.sigma_layer(input_feature)
        intput_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(intput_std)
        input_latent_vector = input_mu + intput_std * input_eps

        # task_feature =self.vae_cl(input_x, output_x).unsqueeze(1).expand(batch_size, 900, 128) # shape을 위해서

        task_feature = self.vae_cl(input_x, output_x)
        task_mu = self.mu_layer(task_feature)
        task_sigma = self.sigma_layer(task_feature)
        task_std = torch.exp(0.5 * task_sigma)
        task_eps = torch.randn_like(task_std)
        task_latent_vector = (task_mu + task_std * task_eps).unsqueeze(1).expand(batch_size, 900, 128)


        # concat_feature = torch.cat((input_latent_vector, task_feature), dim=-1)
        concat_feature = torch.cat((input_latent_vector, task_latent_vector), dim=-1)

        fusion_feature = self.fusion_layer1(concat_feature)
        decoder_output = self.solver_decoder(fusion_feature)

        output = self.solver_proj(decoder_output)

        return output
