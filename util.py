import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, irregularity=None):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

        self.keep = None
        if irregularity is not None:
            print(f"USING IRREGULARITY PROB {irregularity['prob']}")

            if irregularity["mask_load_path"] is not None:
                with open(irregularity["mask_load_path"], "rb") as f:
                    self.keep = pickle.load(f)
                assert self.keep[:, 0].all()
                print(f"LOADED MASK (from {irregularity['mask_load_path']}): {self.keep.shape}, "
                      f"{self.keep.float().mean()}% true")
            else:
                self.keep = torch.rand(len(self.xs), 12) > irregularity["prob"]
                self.keep[:, 0] = True  # Never discard the first time step
                print(f"GENERATED MASK: {self.keep.shape}, {self.keep.float().mean()}% true")

            if irregularity["mask_save_path"] is not None:
                import pickle
                with open(irregularity["mask_save_path"], "wb") as f:
                    pickle.dump(self.keep, f)
                print("DUMPED KEEP MASK TO", irregularity["mask_save_path"])

            self.labelmask = irregularity["labelmask"]
            self.scaler = irregularity["scaler"]
            if self.labelmask:
                print("USING LABEL MASKING")

            if irregularity["mode"] == "MOSTRECENT":
                print("USING MOSTRECENT IRREGULARITY")
                self.irreg_func = most_recent_irreg_func
            elif irregularity["mode"] == "ZERO":
                print("USING ZERO IRREGULARITY")
                self.irreg_func = zero_irreg_func
            elif irregularity["mode"] == "ZEROSCALED":
                print("USING ZERO IRREGULARITY")
                self.irreg_func = lambda x, keep: zero_irreg_func(
                    x, keep, zero_val=self.scaler.transform(0))
            elif irregularity["mode"] == "LINEAR":
                print("USING LINEAR IRREGULARITY")
                self.irreg_func = linear_irreg_func
            else:
                raise ValueError(f"Invalid irregularity mode: {irregularity['mode']}")

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        if self.keep is not None:
            self.keep = self.keep[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]

                # TODO(piyush) remove
                if self.keep is not None:
                    keep = self.keep[start_ind : end_ind, ...]
                    x_i = self.irreg_func(x_i, keep)
                    if self.labelmask:
                        # Make a copy to avoid making permanent changes to the data loader.
                        masked_y = np.empty_like(y_i)
                        masked_y[:] = y_i
                        # masked_y[~keep] = self.scaler.transform(0)
                        masked_y[~keep] = 0.0
                        y_i = masked_y

                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    # TODO(piyush) remove
    irregularity = None
    if "IRREGULARITY" in os.environ:
        irregularity = {
            "mode": os.environ.get("IRREGULARITY", None),
            "prob": float(os.environ.get("PROB", 0.0)),
            "labelmask": "LABELMASK" in os.environ,
            "mask_save_path": os.environ.get("MASKSAVEPATH"),
            "mask_load_path": os.environ.get("MASKLOADPATH"),
            "scaler": scaler,
        }
        print("USING IRREGULARITY")
        print(irregularity)

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size,
                                      irregularity=irregularity)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse



def most_recent_irreg_func(x, keep):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        for t in range(1, x.shape[1]):
            if not keep[i, t]:
                irreg_x[i, t, ...] = irreg_x[i, t - 1, ...]
    return irreg_x

def zero_irreg_func(x, keep, zero_val=0):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        for t in range(1, x.shape[1]):
            if not keep[i, t]:
                irreg_x[i, t, ...] = zero_val
    return irreg_x

def linear_irreg_func(x, keep):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        t = 1
        while t < x.shape[1]:
            if not keep[i, t]:
                start = t
                while t < x.shape[1] and not keep[i, t]:
                    t += 1
                end = t

                irreg_x[i, start : end, ...] = np.array([
                    [
                        np.interp(
                            x=range(start, end), xp=[start - 1, end],
                            fp=[irreg_x[i, start - 1, j1, j2],
                                irreg_x[i, end, j1, j2]])
                        for j2 in range(x.shape[3])
                    ]
                    for j1 in range(x.shape[2])
                ])
            t += 1
    return irreg_x
