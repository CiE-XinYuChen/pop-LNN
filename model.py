import torch
import torch.nn as nn

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.kaiming_uniform_(self.weight_hh)
        nn.init.constant_(self.bias_ih, 0)
        nn.init.constant_(self.bias_hh, 0)

    def forward(self, input, hidden):
        pre_activation = torch.addmm(self.bias_ih, input, self.weight_ih.t()) + torch.addmm(self.bias_hh, hidden, self.weight_hh.t())
        dh = (pre_activation - hidden) / self.tau
        new_hidden = hidden + dh
        return new_hidden


class LTCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LTCNetwork, self).__init__()
        self.ltc_cell = LTCCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = torch.zeros(input.size(0), self.ltc_cell.hidden_size)
        for i in range(input.size(1)):
            hidden = self.ltc_cell(input[:, i, :], hidden)
        output = self.fc(hidden)
        return output, hidden
