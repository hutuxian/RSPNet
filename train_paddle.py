import logging
import arguments
import six
import paddle
import paddle.distributed.spawn as spawn
import numpy as np
import model_paddle as model
import models.resnet as resnet

from arguments import Args
from framework import utils

from datasets.classification import DataLoaderFactoryV3
from framework.config import get_config, save_config
from paddle.distributed import fleet


class Engine:
    def __init__(self, args, cfg, local_rank):
        self.epochs = cfg.get_int('num_epochs')
        self.args = args
        self.current_epoch = 0
        # self.batch_size = self.args.train_batch_size
        self.batch_size = cfg.get_int('batch_size')
        self.batch_size = 64

        fleet.init(is_collective=True)
        self.model = model.MoCoWrapper(cfg).build_moco_diffloss()

        self.data_loader_factory = DataLoaderFactoryV3(cfg)
        self.train_loader = self.data_loader_factory.build(vid=True)

        self.learning_rate = cfg.get_float('optimizer.lr')
        self.loss_lambda = cfg.get_config('loss_lambda')
        self.criterion = model.Loss(
            margin=2.0,
            A=self.loss_lambda.get_float('A'),
            M=self.loss_lambda.get_float('M')
        )

        self.optimizer = paddle.optimizer.Momentum(
            parameters=self.model.parameters(),
            learning_rate=self.learning_rate,
            momentum=cfg.get_float('optimizer.momentum'),
            # dampening=cfg.get_float('optimizer.dampening'),
            weight_decay=cfg.get_float('optimizer.weight_decay'),
            use_nesterov=cfg.get_bool('optimizer.nesterov'),
        )
        self.optimizer = fleet.distributed_optimizer(self.optimizer)
        self.model = fleet.distributed_model(self.model)

    def train_epoch(self):
        loop = 10
        num_iters = len(self.train_loader)
        # clip_q = np.random.random([self.batch_size, 3, 2, 2, 2])
        # clip_k = np.random.random([self.batch_size, 3, 2, 2, 2])
        clip_q = paddle.randn([self.batch_size, 3, 16, 112, 112])
        clip_k = paddle.randn([self.batch_size, 3, 16, 112, 112])
        # for i in range(loop):
        for i, (clip_q, clip_k) in enumerate(self.train_loader):
            output, target, ranking_logits, ranking_target = self.model(clip_q, clip_k)
            loss, loss_A, loss_M = self.criterion(output, target, ranking_logits, ranking_target)

            if not self.args.validate:
                loss.backward()
                self.optimizer.step()
                self.model.clear_gradients()

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1_A = paddle.metric.accuracy(input=output[0], label=target, k=1)
            acc5_A = paddle.metric.accuracy(input=output[0], label=target, k=5)
            
            acc1_M = paddle.metric.accuracy(paddle.concat(ranking_logits, axis=1), label=target, k=1)

            # acc1_A_n = accuracy(output[1], target, k=1)
            # acc5_A_n = accuracy(output[1], target, k=5)

            # if i > 0 and i % self.log_interval == 0:
            if True:
                print("loss: %f, loss_A: %f, loss_M: %f" % (loss, loss_A, loss_M))
                print("acc1_A: %f, acc5_A: %f, acc1_M: %f" % (acc1_A, acc5_A, acc1_M))
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                # logger.info(
                #     f'Train [{epoch}/{self.num_epochs}][{i - 1}/{num_iters}]'
                #     f'\t{self.loss_meter_A}\t{self.top1_meter_A}\t{self.top5_meter_A}\n'
                #     f'{self.loss_meter_M}\t{self.top1_meter_M}\n'
                #     f'{self.top1_meter_A_n}\t{self.top5_meter_A_n}'
                # )


            
    def run(self):
        while self.current_epoch < self.epochs:
            print("train epoch: %d" % (self.current_epoch))
            self.train_epoch()
            self.current_epoch += 1


def main(local_rank = 0):
    # Only single node distributed training is supported

    args = Args.from_args()
    # Run on main process to avoid conflict
    args.resolve_continue()
    args.make_run_dir()
    args.save()
    utils.pack_code(args.run_dir)
    args.parser = None
    cfg = get_config(args)

    print('Local Rank:', local_rank)
    engine = Engine(args, cfg, local_rank=local_rank)
    engine.run()



def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == '__main__':
    main()
