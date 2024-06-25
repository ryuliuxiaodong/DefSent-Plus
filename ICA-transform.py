import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

import sys
from collections import OrderedDict

def load_sys_args():
    argv = sys.argv[1:]
    sys_args = OrderedDict()

    if len(argv) > 0:
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            sys_args[arg_name] = arg_value
    else:
        raise RuntimeError('No input error')

    return sys_args

def ensure_hyperparameters_type(sys_args):
    hyperparameters = OrderedDict()

    try:
        hyperparameters["entry_embed"] = str(sys_args["entry_embed"])
        if "AC" not in hyperparameters["entry_embed"] and "AMP" not in hyperparameters["entry_embed"]:
            raise RuntimeError('name should be either AC_entry_embed_weight.pth or AMP_entry_embed_weight.pth.')
    except:
        raise RuntimeError('the weight matrix to entry embeddings is missing.')
    try:
        hyperparameters["save_path"] = str(sys_args["save_path"])
    except:
        raise RuntimeError('save path is missing.')
    try:
        hyperparameters["max_iteration"] = int(sys_args["max_iteration"])
    except:
        raise RuntimeError('max iteration is missing.')
    try:
        hyperparameters["seed"] = int(sys_args["seed"])
    except:
        hyperparameters["seed"] = 42

    return hyperparameters


def ICA_transform(weight_matirx, max_iteration, seed, save_path):
    embedding_weight_data = torch.load(weight_matirx)
    weight_size = list(embedding_weight_data.size())
    dims = weight_size[1]

    X = embedding_weight_data.to('cpu').detach().numpy().copy()

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    ica = FastICA(n_components=dims, max_iter=max_iteration, tol=1e-4, whiten='unit-variance', fun='logcosh', random_state=seed)
    X_transformed = ica.fit_transform(scaled_X)

    e = torch.from_numpy(X_transformed).clone()
    e = torch.Tensor.float(e)
    e = e * 100

    if "AC" in hyperparameters["entry_embed"]:
        torch.save(e, save_path + "/ICA_AC_entry_embed_weight.pth")
    else:
        torch.save(e, save_path + "/ICA_AMP_entry_embed_weight.pth")


if __name__ == "__main__":
    sys_args = load_sys_args()
    hyperparameters = ensure_hyperparameters_type(sys_args=sys_args)

    ICA_transform(weight_matirx=hyperparameters["entry_embed"],
                  max_iteration=hyperparameters["max_iteration"],
                  seed=hyperparameters["seed"],
                  save_path=hyperparameters["save_path"])