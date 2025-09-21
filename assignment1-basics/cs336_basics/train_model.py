import argparse
import math

import numpy as np
import torch

from cs336_basics.modules.transfomer import TransformerLM
from cs336_basics.train.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.train.utils import get_batch, cross_entropy, save_checkpoint, gradient_clipping


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据与保存路径
    parser.add_argument('--train_path', type=str, default="./data/TinyStoriesV2-GPT4-train_tokens.bin")
    parser.add_argument('--val_path', type=str, default="./data/TinyStoriesV2-GPT4-valid_tokens.bin")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # 模型超参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    # 优化器和训练超参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    return parser.parse_args()

def load_dataset(path, dtype=np.uint16):
    return np.memmap(path, dtype=dtype, mode='r')

def calc_d_ff(d_model: int) -> int:
    raw = (8/3) * d_model
    return math.ceil(raw / 64) * 64

def build_model_optimizer(args):
    d_ff = calc_d_ff(args.d_model)
    model = TransformerLM(d_model=args.d_model,
                          num_heads=args.num_heads,
                          d_ff=d_ff,
                          theta=10000,
                          vocab_size=10000,
                          context_length=512,
                          num_layers=args.num_layers)
    optimizer = AdamW(list(model.parameters()), lr=args.lr)
    return model, optimizer

def train(args):
    model, optimizer = build_model_optimizer(args)
    train_dataset = load_dataset(args.train_path)
    valid_dataset = load_dataset(args.val_path)
    device = "mps"
    model.to(device)
    iteration = 0
    for epoch in range(args.epochs):
        num_batches = len(train_dataset) // args.batch_size
        for _ in range(num_batches):
            x_batch, y_batch = get_batch(train_dataset, batch_size=args.batch_size, context_length=256, device=device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = cross_entropy(y_hat, y_batch)
            loss.backward()

            lr = lr_cosine_schedule(iteration, args.lr , args.lr/100, 1000, 10000)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            gradient_clipping(model.parameters(),1.0)
            optimizer.step()
            iteration += 1
            if iteration % args.log_interval == 0:
                print(f"Epoch {epoch} Iter {iteration} Loss {loss.item():.4f}")
            if iteration % args.save_interval == 0:
                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_{iteration}.pt"
                save_checkpoint(model, optimizer, iteration, checkpoint_path)

        with torch.no_grad():
            x_val, y_val = get_batch(valid_dataset, batch_size=args.batch_size, context_length=512, device=device)
            y_val_hat = model(x_val)
            val_loss = cross_entropy(y_val_hat, y_val)
            print(f"Epoch {epoch} Validation Loss {val_loss.item():.4f}")
if __name__ == "__main__":
    args = parse_args()
    train(args)