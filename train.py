#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
import numpy as np

from ntm.ntm import DualMemoryNTM
from tasks.DualCopyTask import DualCopyTaskParams

LOGGER = logging.getLogger(__name__)


from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.DualCopyTask import DualCopyTaskModelTraining, DualCopyTaskParams
from tasks.CDCLNLITask import CDCLNLITaskModelTraining, CDCLNLITaskParams

LOGGER = logging.getLogger(__name__)

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams)
}

DUAL_TASKS = {
    'copy': (DualCopyTaskModelTraining, DualCopyTaskParams),
    'cdcl-nli': (CDCLNLITaskModelTraining, CDCLNLITaskParams),
}

# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths):
    progress_clean()

    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_batch(net, criterion, optimizer, X, Y):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return loss.item(), cost.item() / batch_size


def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


def train_model(model, args):
    # 检查是否是 CDCL-NLI 任务
    if hasattr(model, 'train_dataloader') and hasattr(model, 'dev_dataloader'):
        # CDCL-NLI 任务使用 epoch 训练
        train_cdcl_nli_model(model, args)
    else:
        # 原有的 batch 训练逻辑
        train_batch_model(model, args)


def train_batch_model(model, args):
    """原有的 batch 训练逻辑"""
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size

    LOGGER.info("Training model for %d batches (batch_size=%d)...",
                num_batches, batch_size)

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()

    for batch_num, x, y in model.dataloader:
        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]

        # Update the progress bar
        progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            progress_clean()
            LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                        batch_num, mean_loss, mean_cost, mean_time)
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, args,
                            batch_num, losses, costs, seq_lengths)

    LOGGER.info("Done training.")


def train_cdcl_nli_model(model, args):
    """CDCL-NLI 任务的 epoch 训练逻辑"""
    from tasks.CDCLNLITask import train_cdcl_nli_batch, evaluate_cdcl_nli
    
    num_epochs = model.params.num_epochs
    batch_size = model.params.batch_size

    LOGGER.info("Training CDCL-NLI model for %d epochs (batch_size=%d)...",
                num_epochs, batch_size)

    for epoch in range(num_epochs):
        model.net.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        start_ms = get_ms()

        for batch_idx, batch in enumerate(model.train_dataloader):
            loss, accuracy = train_cdcl_nli_batch(model, batch)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

            # Update progress bar
            progress_bar(batch_idx + 1, args.report_interval, loss)

            # Report
            if (batch_idx + 1) % args.report_interval == 0:
                mean_loss = total_loss / num_batches
                mean_accuracy = total_accuracy / num_batches
                mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
                progress_clean()
                LOGGER.info("Epoch %d, Batch %d Loss: %.6f Accuracy: %.4f Time: %d ms/batch",
                            epoch + 1, batch_idx + 1, mean_loss, mean_accuracy, mean_time)
                start_ms = get_ms()

        # Epoch summary
        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_accuracy / num_batches
        
        # Evaluate on dev set
        dev_loss, dev_accuracy = evaluate_cdcl_nli(model, model.dev_dataloader)
        
        LOGGER.info("Epoch %d/%d - Train Loss: %.6f, Train Acc: %.4f, Dev Loss: %.6f, Dev Acc: %.4f",
                    epoch + 1, num_epochs, epoch_loss, epoch_accuracy, dev_loss, dev_accuracy)

        # Checkpoint
        if (args.checkpoint_interval != 0) and ((epoch + 1) % args.checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, args,
                            epoch + 1, [epoch_loss], [epoch_accuracy], [num_batches])

    LOGGER.info("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()) + list(DUAL_TASKS.keys()), default='copy',
                        help="Choose the task to train (default: copy)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))
    parser.add_argument('--checkpoint-path', action='store', default='./checkpoints',
                        help="Path for saving checkpoint data (default: './checkpoints')")
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")
    parser.add_argument('--model-type', choices=['ntm', 'dual'], default='ntm',
                        help="Choose model type: ntm or dual-memory")
    parser.add_argument('--num-batches', type=int, default=None,
                        help="Override number of batches to train")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Override batch size")
    parser.add_argument('--long-term-memory-backend', choices=['in-memory', 'neo4j'], default='in-memory',
                        help="Choose long-term memory backend: in-memory or neo4j")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)

    if args.model_type == 'ntm':
        model_cls, params_cls = TASKS[args.task]
    elif args.model_type == 'dual':
        model_cls, params_cls = DUAL_TASKS.get(args.task, (None, None))
        if model_cls is None:
            raise ValueError(f"Task {args.task} not supported with dual memory model.")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    params = params_cls()
    
    # 如果指定了 num_batches，覆盖默认值
    if args.num_batches is not None:
        params = attr.evolve(params, num_batches=args.num_batches)
    
    # 如果指定了 batch_size，覆盖默认值
    if args.batch_size is not None:
        params = attr.evolve(params, batch_size=args.batch_size)
    
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    # 对于双重记忆模型，传递long_term_memory_backend参数
    if args.model_type == 'dual':
        # 从环境变量获取Neo4j配置
        neo4j_config = None
        if args.long_term_memory_backend == "neo4j":
            import os
            neo4j_config = {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password123")
            }
            LOGGER.info("Using Neo4j backend with config: %s", neo4j_config)
        
        model = model_cls(
            params=params,
            long_term_memory_backend=args.long_term_memory_backend,
            neo4j_config=neo4j_config
        )
    else:
        model = model_cls(params=params)
    
    return model


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    init_logging()

    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)

    # 计算总参数数量（包括投影层）
    total_params = model.net.calculate_num_params()
    if hasattr(model, 'projection_layer'):
        total_params += sum(p.numel() for p in model.projection_layer.parameters())
    
    LOGGER.info("Total number of parameters: %d", total_params)
    train_model(model, args)


if __name__ == '__main__':
    main()
