import time
import logging
import warnings
import os
import paddle
import paddle.distributed as dist
from pyhocon import ConfigTree
from paddle import nn
#from torch.utils.tensorboard import SummaryWriter

from paddle.distributed import fleet
import paddle.distributed.fleet.base.role_maker as role_maker

from arguments import Args
from datasets.classification import DataLoaderFactoryV3
from framework import utils
from framework.config import get_config, save_config
from framework.logging import set_logging_basic_config
from framework.meters.average import AverageMeter
from framework.metrics.classification import accuracy
from framework.utils import CheckpointManager, pack_code
from models import ModelFactory

logger = logging.getLogger(__name__)


class EpochContext:
    def __init__(self, engine: 'Engine', name: str, n_crop: int, dataloader, tensorboard_prefix: str):
        self.engine = engine
        self.log_interval = engine.cfg.get_int('log_interval')

        self.n_crop = n_crop
        self.name = name
        self.dataloader = dataloader
        #self.tensorboard_prefix = tensorboard_prefix

        #self.dataloader.set_epoch(self.engine.current_epoch)
        # start dataloader early for better performance
        self.data_iter = iter(dataloader)

        device = self.engine.device
        #self.loss_meter = AverageMeter('Loss', device=device)  # This place displays decimals directly because the loss is relatively large
        #self.top1_meter = AverageMeter('Acc@1', fmt=':6.2f', device=device)
        #self.top5_meter = AverageMeter('Acc@5', fmt=':6.2f', device=device)

    def reshape_clip(self, clip):
        if self.n_crop == 1:
            return clip
        #clip = clip.refine_names('batch', 'channel', 'time', 'height', 'width')
        #crop_len = clip.shape[2] // self.n_crop
        #clip = clip.unflatten('time', [('crop', self.n_crop), ('time', crop_len)])
        #clip = clip.align_to('batch', 'crop', ...)
        #clip = clip.flatten(['batch', 'crop'], 'batch')
        
        b,c,t,h,w=clip.shape
        crop_len = clip.shape[2] // self.n_crop
        clip = clip.reshape((b,c,self.n_crop,crop_len,h,w))
        clip = paddle.transpose(clip,(0,2,1,3,4,5))
        clip = paddle.flatten(clip,start_axis=0,stop_axis=1)       
        return clip

    def average_logits(self, logits):
        if self.n_crop == 1:
            return logits
        #logits = torch.as_tensor(logits.cpu().numpy())
        #logits = logits.refine_names('batch', 'class')
        #num_sample = logits.shape[0] // self.n_crop
        #logits = logits.unflatten('batch', [('batch', num_sample), ('crop', self.n_crop)])
        #logits = logits.mean(dim='crop')
        
        b,cls = logits.shape
        num_sample = logits.shape[0] // self.n_crop
        logits = logits.reshape((num_sample,self.n_crop,cls))
        logits = logits.mean(axis=1)
        return logits

    def meters(self):
        yield self.loss_meter
        yield self.top1_meter
        yield self.top5_meter

    def sync_meters(self):
        for m in self.meters():
            m.sync_distributed()

    def write_tensorboard(self):
        epoch = self.engine.current_epoch
        prefix = self.tensorboard_prefix
        tb = self.engine.summary_writer
        if tb is None:
            return

        tb.add_scalar(
            f'{prefix}/loss', self.loss_meter.avg, epoch
        )
        tb.add_scalar(
            f'{prefix}/acc1', self.top1_meter.avg, epoch
        )
        tb.add_scalar(
            f'{prefix}/acc5', self.top5_meter.avg, epoch
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        #self.write_tensorboard()

    def forward(self):
        logger.info('%s epoch begin.', self.name)
        begin_time = time.perf_counter()
        num_iters = len(self.dataloader)
        print("=============len=======",num_iters)
        for i, ((clip,), target, *others) in enumerate(self.data_iter):
            target = paddle.reshape(target,(-1,1))
            clip = self.reshape_clip(clip)
            output = self.engine.model(clip)
            
            output = self.average_logits(output)
            loss = self.engine.criterion(output, target)

            # This will make tensorboard load very slow. enable if needed
            # if self.engine.summary_writer is not None:
            #     self.engine.summary_writer.add_scalar(f'step/{self.tensorboard_prefix}/loss', loss,
            #         self.engine.current_epoch * num_iters + i)

            if i > 0 and i % self.log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                print("name:{} epoch:{} step:{} loss:{} acc1:{} acc5:{}".format(self.name,self.engine.current_epoch,i,loss.cpu().numpy(),acc1.cpu().numpy(),acc5.cpu().numpy()))

            num_classes = output.shape[1]
            if num_classes >= 5:
                acc1 = paddle.metric.accuracy(output, target,k=1)
                acc5 = paddle.metric.accuracy(output, target,k=5)
            else:
                acc1 = paddle.metric.accuracy(output, target,k=1)
            
            yield loss, output, target.cpu().numpy().flatten().tolist(), acc1, acc5

        end_time = time.perf_counter()
        print("epoch finished. Time: %.2f sec",end_time - begin_time)


class Engine:

    def __init__(self, args: Args, cfg: ConfigTree, local_rank: int, final_validate=False):
        self.args = args
        self.cfg = cfg
        self.local_rank = local_rank

        self.num_epochs = cfg.get_int('num_epochs')
        self.data_loader_factory = DataLoaderFactoryV3(cfg, final_validate)
        self.final_validate = final_validate

        self.device = paddle.CUDAPlace(local_rank)

        self.model_factory = ModelFactory(cfg)
        model_type = cfg.get_string('model_type')
        if model_type == '1stream':
            self.model = self.model_factory.build(local_rank)  # basic model
        elif model_type == 'multitask':
            self.model = self.model_factory.build_multitask_wrapper(local_rank)
        else:
            raise ValueError(f'Unrecognized model_type "{model_type}"')
         
        if not final_validate:
            self.train_loader = self.data_loader_factory.build(
                vid=False,  # need label to gpu
                split='train',
                device=self.device
            )
        self.validate_loader = self.data_loader_factory.build(
            vid=False,
            split='val',
            device=self.device
        )

        if final_validate:
            self.n_crop = cfg.get_int('temporal_transforms.validate.final_n_crop')
        else:
            self.n_crop = cfg.get_int('temporal_transforms.validate.n_crop')

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = self.cfg.get_float('optimizer.lr')
        
        self.schedule_type = self.cfg.get_string('optimizer.schedule')
        if self.schedule_type == "plateau":
            self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=self.learning_rate,
                mode='min',
                patience=self.cfg.get_int('optimizer.patience'),
                verbose=True
            )
        elif self.schedule_type == "multi_step":
            self.scheduler = paddle.optimizer.lr.MultiStepDecay (
                learning_rate=self.learning_rate,
                milestones=self.cfg.get("optimizer.milestones"),
            )
        elif self.schedule_type == "cosine":
            self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=self.learning_rate,
                T_max=self.num_epochs,
                eta_min=self.learning_rate / 1000
            )
        elif self.schedule_type == 'none':
            self.scheduler = paddle.optimizer.lr.LambdaDecay(
                learning_rate=self.learning_rate,
                lr_lambda = lambda epoch: 1,
            )
        else:
            raise ValueError("Unknow schedule type")
         
        
        optimizer_type = self.cfg.get_string('optimizer.type', default='sgd')
        if optimizer_type == 'sgd':
            self.optimizer = paddle.optimizer.Momentum(
                parameters=self.model.parameters(),
                learning_rate=self.scheduler,
                momentum=self.cfg.get_float('optimizer.momentum'),
                #dampening=self.cfg.get_float('optimizer.dampening'),
                weight_decay=self.cfg.get_float('optimizer.weight_decay'),
                use_nesterov=self.cfg.get_bool('optimizer.nesterov'),
            )
        elif optimizer_type == 'adam':
            self.optimizer = paddle.optimizer.Adam(
                self.model.parameters(),
                learning_rate=self.scheduler,
                epsilon=self.cfg.get_float('optimizer.eps'),
            )
        else:
            raise ValueError(f'Unknown optimizer {optimizer_type})')

        if not final_validate:         
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)

        self.arch = cfg.get_string('model.arch')
        
        self.best_acc1 = 0.
        self.current_epoch = 0
        self.next_epoch = None
        logger.info('Engine: n_crop=%d', self.n_crop)

        self.loss_meter = None

    def has_next_epoch(self):
        return not self.final_validate and self.current_epoch < self.num_epochs - 1


    def load_checkpoint(self, checkpoint_path):
        env_path=self.args.experiment_dir
        print(env_path)
        best_param=os.path.join(str(self.args.experiment_dir),"best_fineture_mode.pdparams")
        best_optim=os.path.join(str(self.args.experiment_dir),"best_fineture_optim.pdopt")
        best_sche=os.path.join(str(self.args.experiment_dir),"best_fineture_sche.pdparams")

        print(best_param)
        A=paddle.load(best_param)
        B=paddle.load(best_optim)
        C=paddle.load(best_sche)
        print("========pppppppppp=========")
        self.model.set_state_dict(A)
        #self.optimizer.set_state_dict(B)
        self.scheduler.set_state_dict(C)
        print("===============load params done=============")
        
    def pre_load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')

        logger.info('Loading checkpoint from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])
        logger.info('Checkpoint loaded')

        self.optimizer.load_state_dict(states['optimizer'])
        self.scheduler.load_state_dict(states['scheduler'])
        self.current_epoch = states['epoch']
        self.best_acc1 = states['best_acc1']

    def load_moco_checkpoint(self, checkpoint_path: str):
        # paddle save params directly at present
        moco_state=paddle.load(checkpoint_path)
        prefix = 'encoder_q.'

        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse'] 
        
        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)
               
        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        self.model.set_state_dict(model_state)
    
    def pre_load_moco_checkpoint(self, checkpoint_path: str):
        cp = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in cp and 'arch' in cp:
            logger.info('Loading MoCo checkpoint from %s (epoch %d)', checkpoint_path, cp['epoch'])
            moco_state = cp['model']
            prefix = 'encoder_q.'
        else:
            # This checkpoint is from third-party
            logger.info('Loading third-party model from %s', checkpoint_path)
            if 'state_dict' in cp:
                moco_state = cp['state_dict']
            else:
                # For c3d
                moco_state = cp
                logger.warning('if you are not using c3d sport1m, maybe you use wrong checkpoint')
            if next(iter(moco_state.keys())).startswith('module'):
                prefix = 'module.'
            else:
                prefix = ''

        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse']

        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        msg = self.model.module.load_state_dict(model_state, strict=False)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
        #        set(msg.missing_keys) == {"linear.weight", "linear.bias"} or \
        #        set(msg.missing_keys) == {'head.projection.weight', 'head.projection.bias'} or \
        #        set(msg.missing_keys) == {'new_fc.weight', 'new_fc.bias'},\
        #     msg

        logger.warning(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')

    def train_context(self):
        return EpochContext(
            self, name='Train',
            n_crop=1,
            dataloader=self.train_loader,
            tensorboard_prefix='train')

    def validate_context(self):
        return EpochContext(
            self, name='Validate',
            n_crop=self.n_crop,
            dataloader=self.validate_loader,
            tensorboard_prefix='val')

    def train_epoch(self):
        epoch = self.next_epoch
        if epoch is None:
            epoch = self.train_context()
        self.next_epoch = self.validate_context()

        self.model.train()
        with epoch:
            for loss, *_ in epoch.forward():
                loss.backward()
                self.optimizer.step()
                self.model.clear_gradients()

    def validate_epoch(self):
        epoch = self.next_epoch
        if epoch is None:
            epoch = self.validate_context()
        if self.has_next_epoch():
            self.next_epoch = self.train_context()
        else:
            self.next_epoch = None

        self.model.eval()
        #all_logits = torch.empty(0, device=next(self.model.parameters()).device)
        all_logits = None
        indices = []
        print("===============start valid==============")
        with epoch:
            with paddle.no_grad():
                sum_loss = sum_acc1 = sum_acc5 = 0
                nn = 0
                for loss, logits, others , acc1, acc5 in epoch.forward():
                    #all_logits = torch.cat((all_logits, logits), dim=0)
                    nn+=1
                    sum_loss += loss.cpu().numpy()
                    sum_acc1 += acc1.cpu().numpy()
                    sum_acc5 += acc5.cpu().numpy()
                    if all_logits is None:
                        all_logits=logits
                        continue
                    else:
                        all_logits = paddle.concat((all_logits, logits), axis=0)
            print("Validation finished. avg_loss={} avg_acc1={} avg_acc5={}".format(sum_loss/nn,sum_acc1/nn,sum_acc5/nn))
            
            #logger.info('Validation finished.\n\tLoss = %f\n\tAcc@1 = %.2f%% (%d/%d)\n\tAcc@5 = %.2f%% (%d/%d)',
            #            epoch.loss_meter.avg.item(),
            #            epoch.top1_meter.avg.item(), epoch.top1_meter.sum.item() / 100, epoch.top1_meter.count.item(),
            #            epoch.top5_meter.avg.item(), epoch.top5_meter.sum.item() / 100, epoch.top5_meter.count.item(),
            #            )

        if self.final_validate:
            ds = self.validate_loader.dataset
            if hasattr(ds, 'save_results'):
                assert indices, 'Dataset should return indices to sort logits'
                assert len(indices) == all_logits.shape[0], \
                    f'Length of indices and logits not match. {len(indices)} vs {all_logits.size(0)}'
                #with (self.args.experiment_dir / f'results_{self.local_rank}.json').open('w') as f:
                #    ds.save_results(f, indices, all_logits)
        return sum_acc1/nn

    def run(self):

        num_epochs = 1 if self.args.debug else self.num_epochs

        self.model.train()

        while self.current_epoch < num_epochs:
            logger.info("Current LR:{}".format(self.scheduler.get_lr()))
            self.train_epoch()
            acc1 = self.validate_epoch()
            if self.schedule_type == "plateau":
                self.scheduler.step(self.loss_meter.val.item())
            else:
                self.scheduler.step()

            self.current_epoch += 1

            if fleet.worker_index() == 0:
                is_best = acc1 > self.best_acc1
                self.best_acc1 = max(acc1, self.best_acc1)

                # save_checkpoint({
                #     'epoch': self.current_epoch,
                #     'arch': self.arch,
                #     'model': self.model.module.state_dict(),
                #     'best_acc1': self.best_acc1,
                #     'optimizer': self.optimizer.state_dict(),
                #     'scheduler': self.scheduler.state_dict(),
                # }, is_best, self.args.experiment_dir)
                if is_best:
                    best_param=os.path.join(str(self.args.experiment_dir),"best_fineture_mode.pdparams")
                    best_optim=os.path.join(str(self.args.experiment_dir),"best_fineture_optim.pdopt")
                    best_sche=os.path.join(str(self.args.experiment_dir),"best_fineture_sche.pdparams")
                    paddle.save(self.model.state_dict(),best_param)
                    paddle.save(self.optimizer.state_dict(),best_optim)
                    paddle.save(self.scheduler.state_dict(),best_sche)
                    print("===================save params===================")


def main_worker(local_rank: int, args: Args):
    print('Local Rank:', local_rank)

    #when train and val finished, do the final val
    if False: 
        print("==========final==========")
        cfg = get_config(args)
        engine = Engine(args, cfg, local_rank=local_rank, final_validate=True)
        engine.load_checkpoint("")
        engine.validate_epoch()
        return
    # log in main process only
    if local_rank == 0:
        set_logging_basic_config(args)

    logger.info(f'Args = \n{args}')

    if args.config is not None and args.experiment_dir is not None:
        cfg = get_config(args)
        if local_rank == 0:
            save_config(args, cfg)
            args.save()

        if not args.validate:
            engine = Engine(args, cfg, local_rank=local_rank)
            print("=============ready load data=============")
            if args.load_checkpoint is not None:
                engine.load_checkpoint(args.load_checkpoint)
            elif args.moco_checkpoint is not None:
                engine.load_moco_checkpoint(args.moco_checkpoint)
            engine.run()
            validate_checkpoint = args.experiment_dir / 'model_best.pth.tar'
        else:
            validate_checkpoint = args.load_checkpoint
            if not validate_checkpoint:
                raise ValueError('With "--validate" specified, you should also specify "--load-checkpoint"')
        
        #print("===========final_final=========")
        #if fleet.worker_index() == 0:
        #    logger.info('==============>Doing final validate.')
        #    engine = Engine(args, cfg, local_rank=local_rank, final_validate=True)
        #    engine.load_checkpoint(validate_checkpoint)
        #    engine.validate_epoch()
        #    print("==========complete=============")


def main():
    args = Args.from_args()

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed)

    # run in main process for preventing concurrency conflict
    args.resolve_continue()
    args.make_run_dir()
    args.save()
    pack_code(args.run_dir)

    utils.environment.ulimit_n_max()

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(is_collective=True)
    main_worker(fleet.worker_index(),args)

if __name__ == '__main__':
    main()
    
