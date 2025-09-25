import argparse
import math

import numpy as np
import torch
import time
import datetime
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.utils import get_batch, cross_entropy, gradient_clipping, load_checkpoint


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
    parser.add_argument('--device', type='str', default='cpu')
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
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"模型参数总数: {num_params / 1e6:.2f}M")
    # num_cal = 20000 * args.d_model + args.num_layers*(4*args.d_model**2 + 3*args.d_model*d_ff)
    # print(f"分析计算参数的总数为： {num_cal /1e6:.2f}M")
    return model, optimizer

from tqdm import tqdm

def evaluate(checkpoint_path, val_path):
    args = parse_args()
    model, optimizer = build_model_optimizer(args)
    load_checkpoint(checkpoint_path, model, optimizer)
    val_dataset = load_dataset(val_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss, total_count = 0.0, 0
    num_batches = len(val_dataset) // args.batch_size // 100

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluating", unit="batch"):
            x_batch, y_batch = get_batch(val_dataset, args.batch_size, context_length=128, device=device)
            y_hat = model(x_batch)
            loss = cross_entropy(y_hat, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_count += x_batch.size(0)

    avg_loss = total_loss / total_count
    perplexity = np.exp(avg_loss)
    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

def train(args):
    model, optimizer = build_model_optimizer(args)
    train_dataset = load_dataset(args.train_path)
    valid_dataset = load_dataset(args.val_path)
    model.to(args.device)
    iteration = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()  # 记录epoch开始时间
        num_batches = len(train_dataset) // args.batch_size
        for _ in range(num_batches):
            iter_start = time.time()  # 记录iteration开始时间

            x_batch, y_batch = get_batch(train_dataset, batch_size=args.batch_size, context_length=256, device=args.device)
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
                iter_end = time.time()
                elapsed = iter_end - iter_start
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] Epoch {epoch} Iter {iteration} Loss {loss.item():.4f} Time per iter: {elapsed:.2f}s")

        epoch_end = time.time()
        epoch_elapsed = epoch_end - epoch_start
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Epoch {epoch} completed in {epoch_elapsed:.2f}s")

        # 验证
        with torch.no_grad():
            val_start = time.time()
            x_val, y_val = get_batch(valid_dataset, batch_size=args.batch_size, context_length=512, device=args.device)
            y_val_hat = model(x_val)
            val_loss = cross_entropy(y_val_hat, y_val)
            val_end = time.time()
            val_elapsed = val_end - val_start
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Epoch {epoch} Validation Loss {val_loss.item():.4f} Time: {val_elapsed:.2f}s")
if __name__ == "__main__":
    # args = parse_args()
    # train(args)
    evaluate("/Users/wanghao/PycharmProjects/CS336/assignment1-basics/checkpoints/checkpoint_iter_120000.pt", "./data/TinyStoriesV2-GPT4-valid_tokens.bin")
