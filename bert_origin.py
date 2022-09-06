import os
import random
import numpy as np
import torch
import transformers

import deepspeed

def check_batch_size(inputs, expected_batch_size):
    """Check if it is the last batch."""
    return inputs.shape[0] == expected_batch_size


def load_wikitext2(batch_size):
    """Download (if needed) and load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    max_len = 128

    print("Loading WikiText-2...")
    # TODO: shuffle
    datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_len) * max_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
            for k, t in concatenated_examples.items()
        }
        result["input_ids"] = torch.LongTensor(result["input_ids"])
        result["labels"] = result["input_ids"].clone().numpy()
        if result["input_ids"].shape[0] == 0:
            return result

        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(result["input_ids"].shape)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15) * (result["input_ids"] != 101) * (result["input_ids"] != 102)
        # create selection from mask_arr
        selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
        # apply selection index to inputs.input_ids, adding MASK tokens
        result["input_ids"][:, selection] = 103

        result["input_ids"] = result["input_ids"].numpy()

        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
    )

    lm_datasets["train"].set_format(type="torch", columns=["input_ids", "labels"])
    lm_datasets["validation"].set_format(
        type="torch", columns=["input_ids", "labels"]
    )
    train_loader = torch.utils.data.DataLoader(lm_datasets["train"], batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(lm_datasets["validation"], batch_size=batch_size)

    return train_loader, val_loader


rank = int(os.environ['RANK'])

seq_length = 128
batch_size = 64

torch.manual_seed(2222)
np.random.seed(2222)
random.seed(2222)

class ConvertNLPContext:
    """The context to deal with TOKENIZERS_PARALLELISM."""

    def __init__(self):
        self.tokenizers_parallelism = None

    def __enter__(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            self.tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __exit__(self, ptype, value, trace):
        if self.tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]

with ConvertNLPContext():
    config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
    config.use_cache = False  # Disable model cache to avoid unnecessary model outputs.
    config.vocab_size = 32032
    # config.num_hidden_layers = 1
    model = transformers.BertForMaskedLM(config)

    input_shape = [batch_size, seq_length]

    np_x = np.random.randint(0, 10000, input_shape)
    t_x = torch.tensor(np_x)
    t_y = model(t_x)[0]
    torch.cuda.empty_cache()

model.train()

torch.manual_seed(2222 + rank)
np.random.seed(2222 + rank)
random.seed(2222 + rank)

ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,
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
model, _, _,_ = deepspeed.initialize(model=model,
                                     model_parameters=model.parameters(),
                                     config=ds_config,
                                     dist_init_required=True)

n_epoch = 1
print_period = 1
trainloader, testloader = load_wikitext2(batch_size)

for epoch in range(n_epoch):
    n_iter = 1
    for batch in trainloader:
        if n_iter == 11:
            break

        inputs, labels = batch["input_ids"], batch["labels"]
        # Throw away the last batch if its size is smaller than the expected batch size.
        if not check_batch_size(inputs, batch_size):
            break

        inputs = inputs.to(device="cuda:{}".format(rank))
        labels = labels.to(device="cuda:{}".format(rank))

        t_y = model(inputs)
        t_y = t_y["logits"].view((batch_size * seq_length, -1))
        labels = torch.flatten(labels)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        loss = torch.nn.functional.nll_loss(t_ypred, labels)
        model.backward(loss)
        model.step()

        if torch.distributed.get_rank() == 0 and n_iter % print_period == 0:
            print("[{}, {}] loss: {:.8f}".format(epoch + 1, n_iter, loss.item()))
        n_iter += 1
