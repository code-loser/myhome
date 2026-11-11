import argparse
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")
import os
import hashlib

class IGB260M(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        if self.size == 'experimental':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.size == 'large' or self.size == 'full':
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:

        if self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        # if self.size == 'full':
        #     path = '/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        
        dataset_name = f'IGB260M_{self.args.dataset_size}_cls{self.args.num_classes}'
        if self.args.synthetic:
            dataset_name += '_synthetic'
            print('Synthetic dataset: ', dataset_name)
        if self.args.in_memory:
            dataset_name += '_inmem'
            print('In memory dataset: ', dataset_name)
            
        super().__init__(name=dataset_name, save_dir=os.path.join(self.dir, 'processed'))

    def process(self):
       
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, 
                         in_memory=self.args.in_memory,
                         classes=self.args.num_classes, 
                         synthetic=self.args.synthetic)

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0], node_edges[:, 1]), 
                              num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        if self.args.dataset_size == 'full':
            n_labeled_idx = 227130858 if self.args.num_classes == 19 else 157675969
            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val = int(n_labeled_idx * 0.2)
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        if self.args.dataset_size == 'full':
            test_mask[n_train + n_val:n_labeled_idx] = True
        else:
            test_mask[n_train + n_val:] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class IGB260MDGLDataset_part(DGLDataset):
    def __init__(self, args, load_feat=True, only_feat=False):
        self.dir = args.path
        self.args = args
        self.load_feat = load_feat
        self.only_feat = only_feat

        dataset_name = f'IGB260M_{self.args.dataset_size}_cls{self.args.num_classes}'
        if self.args.synthetic:
            dataset_name += '_synthetic'
            print('Synthetic dataset:', dataset_name)
        if self.args.in_memory:
            dataset_name += '_inmem'
            print('In memory dataset:', dataset_name)

        super().__init__(name=dataset_name, save_dir=os.path.join(self.dir, 'processed'))

    def process(self):
        dataset = IGB260M(
            root=self.dir,
            size=self.args.dataset_size,
            in_memory=self.args.in_memory,
            classes=self.args.num_classes,
            synthetic=self.args.synthetic
        )

        if self.only_feat:
            self.node_features = torch.from_numpy(dataset.paper_feat)
            self.graph = None
            return

        node_edges = torch.from_numpy(dataset.paper_edge)
        num_nodes = 100000000
        self.graph = dgl.graph((node_edges[:, 0], node_edges[:, 1]), num_nodes=num_nodes)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)
        self.graph.ndata['label'] = node_labels
        if self.load_feat:
            node_features = torch.from_numpy(dataset.paper_feat)
            self.graph.ndata['feat'] = node_features

        if self.args.dataset_size == 'full':
            n_labeled_idx = 227130858 if self.args.num_classes == 19 else 157675969
            n_nodes = num_nodes
            n_train = int(n_labeled_idx * 0.6)
            n_val = int(n_labeled_idx * 0.2)
        else:
            n_nodes = num_nodes
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        perm = torch.randperm(n_nodes)
        train_mask[perm[0:n_train]] = True
        val_mask[perm[n_train:n_train+n_val]] = True
        if self.args.dataset_size == 'full':
            test_mask[n_train + n_val:n_labeled_idx] = True
        else:
            test_mask[n_train + n_val:] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        if self.only_feat:
            return self.node_features
        return self.graph

    def __len__(self):
        return 1

    def load_only_features(self):
        dataset = IGB260M(
            root=self.dir,
            size=self.args.dataset_size,
            in_memory=self.args.in_memory,
            classes=self.args.num_classes,
            synthetic=self.args.synthetic
        )
        node_features = torch.from_numpy(dataset.paper_feat)
        return node_features

# class IGB260MDGLDataset(DGLDataset):
#     def __init__(self, args):
#         self.dir = args.path
#         self.args = args
#         super().__init__(name='IGB260MDGLDataset', save_dir=os.path.join(self.dir, 'processed'))
#         # super().__init__(name='IGB260MDGLDataset', save_dir=None)

#     def process(self):
#         # 生成唯一参数哈希标识
#         params = {
#             'dataset_size': self.args.dataset_size,
#             'num_classes': self.args.num_classes,
#             'synthetic': self.args.synthetic,
#             'in_memory': self.args.in_memory,
#         }
#         params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
#         save_path = os.path.join('/data_priv/wangcan/IGB-Datasets/igb/IGB-dataset/', f'processed_graph_{params_hash}.bin')

#         # 检查是否存在已处理的图
#         if os.path.exists(save_path):
#             print("Loading preprocessed graph...")
#             self.graph = dgl.load_graphs(save_path)[0]
#             return

#         # 原始处理流程
#         dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory,
#                           classes=self.args.num_classes, synthetic=self.args.synthetic)

#         node_features = torch.from_numpy(dataset.paper_feat)
#         node_edges = torch.from_numpy(dataset.paper_edge)
#         node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

#         self.graph = dgl.graph((node_edges[:, 0], node_edges[:, 1]), num_nodes=node_features.shape[0])
#         self.graph.ndata['feat'] = node_features
#         self.graph.ndata['label'] = node_labels

#         # 划分掩码
#         if self.args.dataset_size == 'full':
#             if self.args.num_classes == 19:
#                 n_labeled_idx = 227130858
#             else:
#                 n_labeled_idx = 157675969

#             n_nodes = node_features.shape[0]
#             n_train = int(n_labeled_idx * 0.6)
#             n_val = int(n_labeled_idx * 0.2)
#         else:
#             n_nodes = node_features.shape[0]
#             n_train = int(n_nodes * 0.6)
#             n_val = int(n_nodes * 0.2)

#         train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(n_nodes, dtype=torch.bool)

#         train_mask[:n_train] = True
#         val_mask[n_train:n_train + n_val] = True
#         if self.args.dataset_size == 'full':
#             test_mask[n_train + n_val:n_labeled_idx] = True
#         else:
#             test_mask[n_train + n_val:] = True

#         self.graph.ndata['train_mask'] = train_mask
#         self.graph.ndata['val_mask'] = val_mask
#         self.graph.ndata['test_mask'] = test_mask

#         # 保存处理后的图
#         dgl.save_graphs(save_path, self.graph)
#         print(f"Saved processed graph to {save_path}")

#     def __getitem__(self, i):
#         return self.graph

#     def __len__(self):
#         return 1  # 因为只包含一个图


# class IGB260MDGLDataset(DGLDataset):
#     def __init__(self, args):
#         self.dir = args.path
#         self.args = args
#         super().__init__(name='IGB260MDGLDataset')

#     def process(self):
#         dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, \
#             classes=self.args.num_classes, synthetic=self.args.synthetic)

#         node_features = torch.from_numpy(dataset.paper_feat)
#         node_edges = torch.from_numpy(dataset.paper_edge)
#         node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

#         self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
#         self.graph.ndata['feat'] = node_features
#         self.graph.ndata['label'] = node_labels

#         # self.graph = dgl.remove_self_loop(self.graph)
#         # self.graph = dgl.add_self_loop(self.graph)

#         if self.args.dataset_size == 'full':
#             #TODO: Put this is a meta.pt file
#             if self.args.num_classes == 19:
#                 n_labeled_idx = 227130858
#             else:
#                 n_labeled_idx = 157675969

#             n_nodes = node_features.shape[0]
#             n_train = int(n_labeled_idx * 0.6)
#             n_val   = int(n_labeled_idx * 0.2)
            
#             train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#             val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#             test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
#             train_mask[:n_train] = True
#             val_mask[n_train:n_train + n_val] = True
#             test_mask[n_train + n_val:n_labeled_idx] = True
            
#             self.graph.ndata['train_mask'] = train_mask
#             self.graph.ndata['val_mask'] = val_mask
#             self.graph.ndata['test_mask'] = test_mask
#         else:
#             n_nodes = node_features.shape[0]
#             n_train = int(n_nodes * 0.6)
#             n_val   = int(n_nodes * 0.2)
            
#             train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#             val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#             test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
#             train_mask[:n_train] = True
#             val_mask[n_train:n_train + n_val] = True
#             test_mask[n_train + n_val:] = True
            
#             self.graph.ndata['train_mask'] = train_mask
#             self.graph.ndata['val_mask'] = val_mask
#             self.graph.ndata['test_mask'] = test_mask
        
#     def __getitem__(self, i):
#         return self.graph

#     def __len__(self):
#         return len(self.graphs)


class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):

        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))


        if self.args.all_in_edges:
            graph_data = {
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('author', 'written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
                ('institute', 'affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
                ('fos', 'topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0])
            }
        else:
            graph_data = {
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
                ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
                ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
            }
        self.graph = dgl.heterograph(graph_data)     
        self.graph.predict = 'paper'

        if self.args.in_memory:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy')))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'))).to(torch.long)  
            else:
                torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_2K.npy'))).to(torch.long)
        else:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy'), mmap_mode='r'))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'), mmap_mode='r')).to(torch.long)  
            else:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_2K.npy'), mmap_mode='r')).to(torch.long)                

        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        if self.args.in_memory:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author', 'node_feat.npy')))
        else:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        if self.args.in_memory:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy')))       
        else:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        if self.args.in_memory:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy')))       
        else:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



class IGBHeteroDGLDatasetMassive(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))
        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        if self.args.dataset_size == "full":
            num_paper_nodes = 269346174
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 277220883
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

          
        elif self.args.dataset_size == "large":
            num_paper_nodes = 100000000
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 116959896
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'institute', 'node_feat.npy'), mmap_mode='r'))
        fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'fos', 'node_feat.npy'), mmap_mode='r'))
        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': len(institute_node_features), 'fos': len(fos_node_features)}
        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        }
        self.graph = dgl.heterograph(graph_data, num_nodes_dict)  
        self.graph.predict = 'paper'
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/gnndataset', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=1,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--all_in_edges', type=bool, default=True, 
        help="Set to false to use default relation. Set this option to True to use all the relation types in the dataset since DGL samplers require directed in edges.")
    args = parser.parse_args()

    dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]
    print(g)
    homo_g = dgl.to_homogeneous(g)
    print()
    print(homo_g)
    print()
