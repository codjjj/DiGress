from src.diffusion.distributions import DistributionNodes
import src.utils as utils
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=300):
        """统计每张图节点的分布

        Args:
            max_nodes_possible (int, optional): _description_. Defaults to 300.

        Returns:
            [max_node] : -> number of graphs
        """        
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                # 在内层循环中，函数遍历loader中的每个数据项data。使用torch.unique(data.batch, return_counts=True)计算当前批次中每个图的节点数量。data.batch是一个批次的索引向量，其中每个元素表示对应节点所属的图的索引。torch.unique会返回两个张量：unique是批次中不重复的图索引，counts则是每个图中的节点数量。
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        #截断后面用不上的部分
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        """统计数据集trainloader节点的种类分布

        Returns:
            [f_N]: -> 比例
        """        
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        """统计trainloader中的边的种类的分布

        Returns:
            d[f_E+1]: d[0]代表没有边  d[1:]是常规的边的种类
        """        
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        """统计所有原子的化合价

        Args:
            max_n_nodes (_type_): _description_

        Returns:
            valenies : 原子化合价的分布,几价的分别有几个
        """        
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                #当前的atom连边的edge_attr
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                #得到总的化合价
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        #归一化
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """计算模型的输入X,E,y的维度和输出的维度

        Args:
            datamodule (_type_): _description_
            extra_features (_type_): _description_
            domain_features (_type_): _description_
        """        
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1) 
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}
