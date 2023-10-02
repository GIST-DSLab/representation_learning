import torch
import torch.nn as nn
from .phase0_model import *

class vae_classifier_v1(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_v1(model_file)
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128
        self.last_parameter_size = 128
        self.num_categories = 16

        self.fusion_layer1 = nn.Linear(2*5*128*900, self.first_layer_parameter_size)
        # self.fusion_layer1 = nn.Linear(128, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)

        self.rnn = nn.RNN(self.first_layer_parameter_size, self.first_layer_parameter_size, 3, nonlinearity='relu', dropout=0.1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.batch_norm = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.layer_norm = nn.LayerNorm(self.first_layer_parameter_size)



        #TODO Modulelist와 for문으로 다시 작성하기
        self.move_layer = nn.Linear(self.last_parameter_size, 1)
        self.color_layer = nn.Linear(self.last_parameter_size, 1)
        self.object_layer = nn.Linear(self.last_parameter_size, 1)
        self.pattern_layer = nn.Linear(self.last_parameter_size, 1)
        self.count_layer = nn.Linear(self.last_parameter_size, 1)
        self.crop_layer = nn.Linear(self.last_parameter_size, 1)
        self.boundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.center_layer = nn.Linear(self.last_parameter_size, 1)
        self.resie_layer = nn.Linear(self.last_parameter_size, 1)
        self.inside_layer = nn.Linear(self.last_parameter_size, 1)
        self.outside_layer = nn.Linear(self.last_parameter_size, 1)
        self.remove_layer = nn.Linear(self.last_parameter_size, 1)
        self.copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.position_layer = nn.Linear(self.last_parameter_size, 1)
        self.direction_layer = nn.Linear(self.last_parameter_size, 1)
        self.bitwise_layer = nn.Linear(self.last_parameter_size, 1)
        self.connect_layer = nn.Linear(self.last_parameter_size, 1)
        self.order_layer = nn.Linear(self.last_parameter_size, 1)
        self.combine_layer = nn.Linear(self.last_parameter_size, 1)
        self.fill_layer = nn.Linear(self.last_parameter_size, 1)


        self.AboveBelow_layer = nn.Linear(self.last_parameter_size, 1)
        self.Center_layer = nn.Linear(self.last_parameter_size, 1)
        self.CleanUp_layer = nn.Linear(self.last_parameter_size, 1)
        self.CompleteShape_layer = nn.Linear(self.last_parameter_size, 1)
        self.Copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.Count_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtendToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtractObjects_layer = nn.Linear(self.last_parameter_size, 1)
        self.FilledNotFilled_layer = nn.Linear(self.last_parameter_size, 1)
        self.HorizontalVertical_layer = nn.Linear(self.last_parameter_size, 1)
        self.InsideOutside_layer = nn.Linear(self.last_parameter_size, 1)
        self.MoveToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.Order_layer = nn.Linear(self.last_parameter_size, 1)
        self.SameDifferent_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom2D_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom3D_layer = nn.Linear(self.last_parameter_size, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]

        input_padding_position = torch.where(input_x == 0, False, True)
        output_padding_position = torch.where(output_x == 0, False, True)

        if len(input_x.shape) > 3:
            batch_size = input_x.shape[0]

            embed_input = self.autoencoder.embedding(input_x.reshape(batch_size, input_x.shape[1]*input_x.shape[2]*input_x.shape[3]).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(batch_size, output_x.shape[1]*output_x.shape[2]* output_x.shape[3]).to(torch.long))
        else:
            embed_input = self.autoencoder.embedding(input_x.reshape(-1, input_x.shape[1]*input_x.shape[2]*input_x.shape[3]).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(-1, output_x.shape[1]*output_x.shape[2]* output_x.shape[3]).to(torch.long))

        embed_input[input_padding_position.reshape(batch_size, -1)] = 0
        embed_output[output_padding_position.reshape(batch_size, -1)] = 0

        input_feature = self.autoencoder.encoder(embed_input)
        output_feature = self.autoencoder.encoder(embed_output)

        input_mu = self.autoencoder.mu_layer(input_feature)
        input_sigma = self.autoencoder.sigma_layer(input_feature)
        intput_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(intput_std)
        input_latent_vector = input_mu + intput_std * input_eps

        output_mu = self.autoencoder.mu_layer(output_feature)
        output_sigma = self.autoencoder.sigma_layer(output_feature)
        output_std = torch.exp(0.5 * output_sigma)
        output_eps = torch.randn_like(output_std)
        output_latent_vector = output_mu + output_std * output_eps

        # concat_feature = torch.cat((input_latent_vector, output_latent_vector),dim=-1)
        concat_feature = torch.cat((input_latent_vector, output_latent_vector), dim=1)
        #
        # fusion_feature = self.fusion_layer1(concat_feature)
        fusion_feature = self.fusion_layer1(concat_feature.reshape(batch_size, -1))
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        # fusion_feature, _ = self.rnn(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        pre_fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        # ===================== 경계선 ======================#

        AboveBelow_output = self.AboveBelow_layer(fusion_feature)
        Center_output = self.Center_layer(fusion_feature)
        CleanUp_output = self.CleanUp_layer(fusion_feature)
        CompleteShape_output = self.CompleteShape_layer(fusion_feature)
        Copy_output = self.Copy_layer(fusion_feature)
        Count_layer = self.Count_layer(fusion_feature)
        ExtendToBoundary_output = self.ExtendToBoundary_layer(fusion_feature)
        ExtractObjects_output = self.ExtractObjects_layer(fusion_feature)
        FilledNotFilled_output = self.FilledNotFilled_layer(fusion_feature)
        HorizontalVertical_output = self.HorizontalVertical_layer(fusion_feature)
        InsideOutside_output = self.InsideOutside_layer(fusion_feature)
        MoveToBoundary_output = self.MoveToBoundary_layer(fusion_feature)
        Order_output = self.Order_layer(fusion_feature)
        SameDifferent_output = self.SameDifferent_layer(fusion_feature)
        TopBottom2D_output = self.TopBottom2D_layer(fusion_feature)
        TopBottom3D_output = self.TopBottom3D_layer(fusion_feature)


        output = torch.stack([AboveBelow_output, Center_output, CleanUp_output, CompleteShape_output, Copy_output, Count_layer, ExtendToBoundary_output, ExtractObjects_output, FilledNotFilled_output, HorizontalVertical_output, InsideOutside_output, MoveToBoundary_output, Order_output, SameDifferent_output, TopBottom2D_output, TopBottom3D_output])

        return output

class vae_classifier_v2(nn.Module):
    def __init__(self, model_file):
        super().__init__()

        self.autoencoder = vae_linear_origin(model_file)
        self.autoencoder.load_state_dict(torch.load(model_file))
        self.auto_encoder_freeze()

        self.first_layer_parameter_size = 128
        self.second_layer_parameter_size = 128
        self.last_parameter_size = 128
        self.num_categories = 16

        self.fusion_layer1 = nn.Linear(2*5*128*900, self.first_layer_parameter_size)
        # self.fusion_layer1 = nn.Linear(128, self.first_layer_parameter_size)
        self.fusion_layer2 = nn.Linear(self.first_layer_parameter_size, self.second_layer_parameter_size)

        self.rnn = nn.RNN(self.first_layer_parameter_size, self.first_layer_parameter_size, 3, nonlinearity='relu', dropout=0.1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.batch_norm = nn.BatchNorm1d(self.first_layer_parameter_size)
        self.layer_norm = nn.LayerNorm(self.first_layer_parameter_size)



        #TODO Modulelist와 for문으로 다시 작성하기
        self.move_layer = nn.Linear(self.last_parameter_size, 1)
        self.color_layer = nn.Linear(self.last_parameter_size, 1)
        self.object_layer = nn.Linear(self.last_parameter_size, 1)
        self.pattern_layer = nn.Linear(self.last_parameter_size, 1)
        self.count_layer = nn.Linear(self.last_parameter_size, 1)
        self.crop_layer = nn.Linear(self.last_parameter_size, 1)
        self.boundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.center_layer = nn.Linear(self.last_parameter_size, 1)
        self.resie_layer = nn.Linear(self.last_parameter_size, 1)
        self.inside_layer = nn.Linear(self.last_parameter_size, 1)
        self.outside_layer = nn.Linear(self.last_parameter_size, 1)
        self.remove_layer = nn.Linear(self.last_parameter_size, 1)
        self.copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.position_layer = nn.Linear(self.last_parameter_size, 1)
        self.direction_layer = nn.Linear(self.last_parameter_size, 1)
        self.bitwise_layer = nn.Linear(self.last_parameter_size, 1)
        self.connect_layer = nn.Linear(self.last_parameter_size, 1)
        self.order_layer = nn.Linear(self.last_parameter_size, 1)
        self.combine_layer = nn.Linear(self.last_parameter_size, 1)
        self.fill_layer = nn.Linear(self.last_parameter_size, 1)


        self.AboveBelow_layer = nn.Linear(self.last_parameter_size, 1)
        self.Center_layer = nn.Linear(self.last_parameter_size, 1)
        self.CleanUp_layer = nn.Linear(self.last_parameter_size, 1)
        self.CompleteShape_layer = nn.Linear(self.last_parameter_size, 1)
        self.Copy_layer = nn.Linear(self.last_parameter_size, 1)
        self.Count_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtendToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.ExtractObjects_layer = nn.Linear(self.last_parameter_size, 1)
        self.FilledNotFilled_layer = nn.Linear(self.last_parameter_size, 1)
        self.HorizontalVertical_layer = nn.Linear(self.last_parameter_size, 1)
        self.InsideOutside_layer = nn.Linear(self.last_parameter_size, 1)
        self.MoveToBoundary_layer = nn.Linear(self.last_parameter_size, 1)
        self.Order_layer = nn.Linear(self.last_parameter_size, 1)
        self.SameDifferent_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom2D_layer = nn.Linear(self.last_parameter_size, 1)
        self.TopBottom3D_layer = nn.Linear(self.last_parameter_size, 1)

    def auto_encoder_freeze(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input_x, output_x):
        batch_size = input_x.shape[0]

        input_padding_position = torch.where(input_x == 0, False, True)
        output_padding_position = torch.where(output_x == 0, False, True)

        if len(input_x.shape) > 3:
            batch_size = input_x.shape[0]

            embed_input = self.autoencoder.embedding(input_x.reshape(batch_size, input_x.shape[1]*input_x.shape[2]*input_x.shape[3]).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(batch_size, output_x.shape[1]*output_x.shape[2]* output_x.shape[3]).to(torch.long))
        else:
            embed_input = self.autoencoder.embedding(input_x.reshape(-1, input_x.shape[1]*input_x.shape[2]*input_x.shape[3]).to(torch.long))
            embed_output = self.autoencoder.embedding(output_x.reshape(-1, output_x.shape[1]*output_x.shape[2]* output_x.shape[3]).to(torch.long))

        embed_input[input_padding_position.reshape(batch_size, -1)] = 0
        embed_output[output_padding_position.reshape(batch_size, -1)] = 0

        input_feature = self.autoencoder.encoder(embed_input)
        output_feature = self.autoencoder.encoder(embed_output)

        input_mu = self.autoencoder.mu_layer(input_feature)
        input_sigma = self.autoencoder.sigma_layer(input_feature)
        intput_std = torch.exp(0.5 * input_sigma)
        input_eps = torch.randn_like(intput_std)
        input_latent_vector = input_mu + intput_std * input_eps

        output_mu = self.autoencoder.mu_layer(output_feature)
        output_sigma = self.autoencoder.sigma_layer(output_feature)
        output_std = torch.exp(0.5 * output_sigma)
        output_eps = torch.randn_like(output_std)
        output_latent_vector = output_mu + output_std * output_eps

        # concat_feature = torch.cat((input_latent_vector, output_latent_vector),dim=-1)
        concat_feature = torch.cat((input_latent_vector, output_latent_vector), dim=1)
        #
        # fusion_feature = self.fusion_layer1(concat_feature)
        fusion_feature = self.fusion_layer1(concat_feature.reshape(batch_size, -1))
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        # fusion_feature, _ = self.rnn(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        # fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        pre_fusion_feature = self.leaky_relu(fusion_feature)

        fusion_feature = self.fusion_layer2(fusion_feature)
        # fusion_feature = self.dropout(fusion_feature)
        fusion_feature = self.batch_norm(fusion_feature)
        fusion_feature += pre_fusion_feature
        # fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.leaky_relu(fusion_feature)

        # ===================== 경계선 ======================#

        AboveBelow_output = self.AboveBelow_layer(fusion_feature)
        Center_output = self.Center_layer(fusion_feature)
        CleanUp_output = self.CleanUp_layer(fusion_feature)
        CompleteShape_output = self.CompleteShape_layer(fusion_feature)
        Copy_output = self.Copy_layer(fusion_feature)
        Count_layer = self.Count_layer(fusion_feature)
        ExtendToBoundary_output = self.ExtendToBoundary_layer(fusion_feature)
        ExtractObjects_output = self.ExtractObjects_layer(fusion_feature)
        FilledNotFilled_output = self.FilledNotFilled_layer(fusion_feature)
        HorizontalVertical_output = self.HorizontalVertical_layer(fusion_feature)
        InsideOutside_output = self.InsideOutside_layer(fusion_feature)
        MoveToBoundary_output = self.MoveToBoundary_layer(fusion_feature)
        Order_output = self.Order_layer(fusion_feature)
        SameDifferent_output = self.SameDifferent_layer(fusion_feature)
        TopBottom2D_output = self.TopBottom2D_layer(fusion_feature)
        TopBottom3D_output = self.TopBottom3D_layer(fusion_feature)


        output = torch.stack([AboveBelow_output, Center_output, CleanUp_output, CompleteShape_output, Copy_output, Count_layer, ExtendToBoundary_output, ExtractObjects_output, FilledNotFilled_output, HorizontalVertical_output, InsideOutside_output, MoveToBoundary_output, Order_output, SameDifferent_output, TopBottom2D_output, TopBottom3D_output])

        return output

