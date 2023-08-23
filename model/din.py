import sys

sys.path.append("..")

import torch
import torch.nn as nn
from basic.layers import EmbeddingLayer, MLP

class DIN (nn.Module):
    """

    """
    #设置模型的结构和参数
    def __init__(self, features, history_features, target_features, mlp_params, attention_mlp_params ):
        super().__init__()
        self.features = features # 代表用户的配置文件特征和上下文特征
        self.history_features = history_features # 代表用户行为序列特征，物品id,商品id
        self.target_features = target_features # 将与历史特征一起执行目标注意力的目标特征
        self.num_history_features = len(history_features)

        self.all_dims =sum([fea.embed_dim for fea in features +history_features +target_features])
        self.embedding = EmbeddingLayer(features + history_features + target_features)

        #ModuleList被用来创建多个ActivationUnit类
        self.attention_layers = nn.ModuleList(
            [ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features]
        )

        #dim:维度列表 activation:激活函数名称 dropout:dropout率 use_softmax: 是否对注意力权重应用softmax
        self.mlp = MLP(self.all_dims, activation="dice", **mlp_params)

        #执行DIN模型的前向传播，接受一个参数x，表示输入数据
    def forward(self,x):
        embed_x_features = self.embedding(x, self.features) #(batch_size, num_features, emb_dim)
        embed_x_history = self.embedding(x, self.history_features) #(batch_size,num_history_features,seq_length,emb_dim)
        embed_x_target = self.embedding(x, self.target_features) #(batch_size, num_target_features, emb_dim)

        #对历史特征和目标特征应用目标注意力，将结果存储在‘attention_pooling’
        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:,i,:,:], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1)) #(batch_size,1 ,emb_dim)
        attention_pooling = torch.cat(attention_pooling,dim=1) #(batch_size, num_history_features, emb_dim)

        mlp_in = torch.cat([
            attention_pooling.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
        ],dim=1)# (batch_size,N)

        y=self.mlp(mlp_in)

        return torch.sigmoid(y.squeeze(1))

#计算历史特征与目标特征之间的注意力权重，DIN类包含了多个ActivationUnit类模块用于对多个历史特征和目标特征执行注意力
class ActivationUnit (nn.Module):
    """
        Activation Unit Target Attention method
    """
    def __init__(self, emb_dim, dims=None, activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        if dims is None:
            dims = [36]
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MLP(4*self.emb_dim, dims=dims, activation=activation)

    def forward(self, history, target):
        seq_length = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_length, -1) #batch_size,seq_length,emb_dim
        att_input = torch.cat([target, history, target - history, target*history])
        att_weight = self.attention(att_input.view(-1, 4*self.emb_dim))
        att_weight = att_weight.view(-1, seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)
        output = (att_weight.unsqueeze(-1)*history).sum(dim=1)
        return output
