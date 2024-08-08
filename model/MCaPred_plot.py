# @Author  : hdm
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer, CrossNet
from deepctr_torch.models.deepfm import *
from model.basemodel_plot import *
from layer.MOELayer import MOELayer
class MCaPred(MyBaseModel):

    def __init__(self, dnn_feature_columns, num_tasks, tasks, dense_num, sparse_num, emb_size,
                 num_experts=7, cross_num=1, dnn_hidden_units=(128, 128),l2_reg_embedding=0.01, l2_reg=0.01, l2_cross=0.01, dr=0, seed=1024, device='cpu'):
        super(MCaPred, self).__init__(linear_feature_columns=dnn_feature_columns, dnn_feature_columns=dnn_feature_columns,
                                       num_tasks=num_tasks, task=tasks,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
        if num_tasks < 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.tasks = tasks
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.sparse_num = sparse_num
        self.cross_num = cross_num
        self.dense_num = dense_num
        self.emb_size = emb_size
        
        self.moe_layer = MOELayer(self.compute_input_dim(dnn_feature_columns), num_tasks, num_experts, dnn_hidden_units, dr, device)

        self.cross_asv = SelfAttentionCrossNet(dense_num, hidden_size=64)
        self.hidden_size = self.cross_asv.hidden_size
        # self.add_regularization_weight(self.cross_asv.kernels, l2 = l2_cross)
        
        if self.compute_input_dim(dnn_feature_columns)-dense_num>0:
            self.cross_host = CrossNet(self.compute_input_dim(dnn_feature_columns)-dense_num, layer_num = cross_num,
                                       parameterization='matrix',device=device)
            self.add_regularization_weight(self.cross_host.kernels, l2 = l2_cross)

        
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.moe_layer.expert_network.named_parameters()), l2=l2_reg)
        tower_dim = dnn_hidden_units[-1] + self.hidden_size + self.compute_input_dim(dnn_feature_columns) - dense_num
        self.tower_network = nn.ModuleList([nn.Linear(tower_dim, 1, bias=False).to(device) for _ in range(num_tasks)])
        
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        
        self.to(device)

    def forward(self, X):
        
        sparse_embedding_list = [X[:,i*self.embedding_size:i*self.emb_size+self.emb_size].reshape(-1,1,self.emb_size) for i in range(self.sparse_num)]
        
        dense_value_list = []
        dense_input = X[:,self.sparse_num*self.emb_size:].transpose(0,1).reshape(self.dense_num,-1,1)

        dense_value_list = [x for x in dense_input]
        
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        cross_out1 = self.cross_asv(dnn_input[:,-1*self.dense_num:])        
        moe_out, self.gate_list = self.moe_layer(dnn_input)
        if len(sparse_embedding_list)!=0:
            cross_out2 = self.cross_host(dnn_input[:,:-1*self.dense_num])
        
        task_outputs = []
        for i in range(self.num_tasks):  
            if len(sparse_embedding_list) != 0:
                tower_input = torch.cat((moe_out[i], cross_out1, cross_out2),1)
            else:
                tower_input = torch.cat((moe_out[i], cross_out1),1)
            
            logit = self.tower_network[i](tower_input)
            y_pred = self.out[i](logit)
            task_outputs.append(y_pred)

        task_outputs = torch.cat(task_outputs, -1)
        
        return task_outputs

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.fc = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # x = self.fc(x)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.weight * x + self.bias


class SelfAttentionCrossNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dropout_prob=0.1):
        super(SelfAttentionCrossNet, self).__init__()

        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)

    def forward(self, x):

        query_layer = self.query(x)
        key_layer = self.key(x)
        value_layer = self.value(x)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.hidden_size ** 0.5)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        output = torch.sigmoid(hidden_states)

        return output
