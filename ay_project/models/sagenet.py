import torch as th
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.dataloading
import torch.nn.functional as F


class SAGENet(th.nn.Module):
   
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGENet, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = dglnn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = dglnn.SAGEConv(h_feats, num_classes, 'mean')
        self.dropout = th.nn.Dropout(0.5)
        self.h_feats = h_feats

    def forward(self, blocks, x):
        h = x
        h = self.conv1(blocks[0], h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(blocks[1], h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv3(blocks[2], h)
        return h

    def inference(self, g, x, batch_size, device):
        nodes = th.arange(g.num_nodes())

        for i, layer in enumerate(self.layers):
            print(f"[Layer {i}] Computing inference with full aggregation")

            if i == len(self.layers) - 1:
                out_dim = self.n_classes
            else:
                out_dim = self.n_hidden

            y = th.zeros((g.num_nodes(), out_dim), dtype=th.float32, device='cpu')

            sampler = dgl.dataloading.NeighborSampler([-1])

            dataloader = dgl.dataloading.DataLoader(
                g,
                nodes,  
                sampler,
                batch_size=batch_size,
                shuffle=False, 
                drop_last=False,
                num_workers=0 
            )

            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                
                
                h_dst = h[:block.number_of_dst_nodes()]

                
                out = layer(block, (h, h_dst))
                
                
                if i != len(self.layers) - 1:
                    out = self.activation(out)
                    out = self.dropout(out)

                
                y[output_nodes] = out.cpu()

            x = y
            
        return x