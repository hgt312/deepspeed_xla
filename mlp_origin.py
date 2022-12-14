import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = x
        hidden = self.linear(hidden)
        return self.cross_entropy_loss(hidden, y)


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=0)
    args = parser.parse_args()  # args=''

    # config_dict["zero_optimization"]["stage"] = args.zero
    # print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)


rank = int(os.environ['RANK'])

config_dict = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
            "torch_adam": True,
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000
    },
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": False,
        "stage3_param_persistence_threshold": 0
    }
}
args = get_args('/tmp/', config_dict)
hidden_dim = 4

print('seed:', 2222)
torch.random.manual_seed(2222)
model = SimpleModel(hidden_dim)
torch.random.manual_seed(2222 + rank)

model, _, _,_ = deepspeed.initialize(args=args,
                                     model=model,
                                     model_parameters=model.parameters(),
                                     dist_init_required=True)


def print_params(tag, model):
    if torch.distributed.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))


data_loader = get_data_loader(model=model,
                              total_samples=1000,
                              hidden_dim=hidden_dim,
                              device="cpu")
# print_params('pre-train', model)
for n, batch in enumerate(data_loader):
    x = batch[0].to(device=model.device)
    y = batch[1].to(device=model.device)
    loss = model(x, y)
    if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
    model.backward(loss)
    model.step()
    # print_params('step={}'.format(n), model)
    if n == 10: break
