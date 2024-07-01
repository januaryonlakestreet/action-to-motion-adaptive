import torch
import torch.nn as nn


class GaussianGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(GaussianGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            h_in = self.hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, h_in


class adaptive_instance_norm(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(channels)
        self.scale_transform = nn.Linear(w_dim, channels)
        self.shift_transform = nn.Linear(w_dim, channels)

        self.std_embedded = nn.Linear(115,channels)
        self.mean_embedded = nn.Linear(115,channels)


    def forward(self, content_features, w):

        scale_tensor = self.scale_transform(w)[:, :]
        shift_tensor = self.shift_transform(w)[:, :]

        w_std = w.std(dim=0)
        w_mean = w.mean(dim=0)

        w_std = self.std_embedded(w_std)
        w_mean = self.mean_embedded(w_mean)

        normalized_image = self.instance_norm(content_features)

        normalized_image = w_std * normalized_image + w_mean




        transformed_image = scale_tensor * normalized_image + shift_tensor
        return transformed_image

class DecoderGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(DecoderGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.adaptive = adaptive_instance_norm(128,115)
        self.style_embed = nn.Linear(input_size,hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def forward(self, inputs,style):

        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded

        embedded_style = self.style_embed(style.view(-1, self.input_size))
        s_in = embedded_style

        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            self.hidden[i] = self.adaptive(self.hidden[i],style)
            h_in = self.hidden[i]

        return self.output(h_in), h_in
#self.adaptive(self.hidden, torch.full((128, 128), 1.0).float().to('cuda'))
# generator with Lie algbra parameters, root joint has no rotations
class DecoderGRULie(DecoderGRU):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(DecoderGRULie, self).__init__(input_size,
                                            output_size,
                                            hidden_size,
                                            n_layers,
                                            batch_size,
                                            device)
        self.output_lie = nn.Linear(output_size - 3, output_size - 3)
        self.PI = 3.1415926

    def forward(self, inputs):
        hidden_output, h_mid = super(DecoderGRULie, self).forward(inputs)
        root_trans = hidden_output[..., :3]
        lie_hid = hidden_output[..., 3:]
        lie_hid = torch.tanh(lie_hid)
        lie_out = self.output_lie(lie_hid)
        lie_out = torch.tanh(lie_out) * self.PI
        output = torch.cat((root_trans, lie_out), dim=-1)
        return output, h_mid
