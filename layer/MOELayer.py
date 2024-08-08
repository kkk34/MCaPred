import torch
import torch.nn as nn
from deepctr_torch.layers import DNN

class MOELayer(nn.Module):
    def __init__(self, input_dim, num_tasks, num_experts, dnn_hidden_units, dr, device):
        super(MOELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts     # expert numbers
        self.num_tasks = num_tasks         # task numbers
        self.dnn_hidden_units = dnn_hidden_units  # hidden numbers
        self.expert_network = nn.ModuleList([DNN(input_dim, dnn_hidden_units,
               activation='relu', l2_reg=0.1, dropout_rate=dr, use_bn=True,
               init_std=0.0001, device=device) for _ in range(self.num_experts)])

        # Calculate expert weights
        self.gate_list = {}   # Gating weight for each task
        for g in range(self.num_tasks):
            self.gate_list[g] = []
        
        self.d = {}   # Save gated learnable parameters
        for i in range(self.num_tasks):
            # Generate a tensor containing num_experts random numbers that will serve as initialized gating weights
            self.d['gate_'+str(i)] = nn.Parameter(torch.rand(self.num_experts), requires_grad=True)
            # Add the gating weight parameter to the parameters of the model
        self.gate = nn.ParameterDict(self.d)


    def forward(self, inputs):
        # print(inputs.shape)
        expert_list = []
        for i in range(self.num_experts):
            expert_out = self.expert_network[i](inputs)
            expert_list.append(expert_out)

        final_expert = torch.stack(expert_list, 2)  # experts output
        # Combination of experts output
        outputs = []
        for i in range(self.num_tasks):
            self.gate_list[i] = self.gate['gate_'+str(i)].softmax(0)
            out_ = final_expert*self.gate_list[i]       
            out = torch.sum(out_, 2)   # Weighted sum
            outputs.append(out)
                    
        return outputs, self.gate_list