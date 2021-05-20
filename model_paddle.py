import paddle
import paddle.nn as nn
import random
import multitask_wrapper_paddle
from models_paddle import get_model_class
# import logger

class MoCoWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build_moco_diffloss(self):
        moco_dim = self.cfg.get_int('moco.dim')
        moco_t = self.cfg.get_float('moco.t')
        moco_k = self.cfg.get_int('moco.k')
        moco_m = self.cfg.get_float('moco.m')
        moco_fc_type = self.cfg.get_string('moco.fc_type')
        moco_diff_speed = self.cfg.get_list('moco.diff_speed')
        backbone = self.cfg.get_config('model').get_string("arch")
        print("backbone: ", backbone)

        base_model_class = get_model_class(backbone)

        def model_class(num_classes=128):
            model = multitask_wrapper_paddle.MultiTaskWrapper(
                base_model_class,
                num_classes=num_classes,
                fc_type=moco_fc_type,
                finetune=False,
                groups=1,
            )
            return model

        model = MoCoDiffLossTwoFc(
            model_class,
            dim=moco_dim,
            K=moco_k,
            m=moco_m,
            T=moco_t,
            diff_speed=moco_diff_speed,
        )
        # model.cuda()
        # model = nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=[dist.get_rank()],
        #     find_unused_parameters=True,
        # )

        return model

class Loss(nn.Layer):
    def __init__(self, margin=1.0, A=1.0, M=1.0):
        super(Loss, self).__init__()
        self.A = A
        self.M = M
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._margin_ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, output, target, ranking_logits, ranking_target):
        ce1 = self._cross_entropy_loss(output[0], target)
        ce2 = self._cross_entropy_loss(output[1], target)
        cast = paddle.tensor.cast(ranking_logits[1], 'float32')
        ranking = self._margin_ranking_loss(ranking_logits[0], ranking_logits[1], cast)
        loss = self.A * (ce1 + ce2) + self.M * ranking
        return loss, ce1 + ce2, ranking

class MoCoDiffLossTwoFc(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, diff_speed=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoDiffLossTwoFc, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.diff_speed = diff_speed
        # logger.warning('Using diffspeed: %s', self.diff_speed)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.trainable = False  # not update by gradient
            # param_k.data.copy_(param_q.data)  # initialize
            # param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = nn.functional.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))
        self.alpha = 0.5	

        assert self.diff_speed is not None, "This branch is for diff speed"

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.set_value(param_k * self.m + param_q * (1. - self.m))
            # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1,0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        if paddle.distributed.get_world_size() > 1:
            # print('forward worlder size', paddle.distributed.get_world_size())
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]
        # return paddle.gather(x_gather, idx_this), idx_unshuffle
        return paddle.index_select(x_gather, idx_this), idx_unshuffle
        # return x_gather[idx_this], idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return paddle.index_select(x_gather, idx_this)
        # return x_gather[idx_this]

    @paddle.no_grad()
    def _forward_encoder_k(self, im_k):
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k_A, k_M = self.encoder_k(im_k)  # keys: NxC
        # k_A = self.encoder_q(im_k)
        # k_M = self.encoder_q(im_k)

        # undo shuffle
        k_A = self._batch_unshuffle_ddp(k_A, idx_unshuffle)
        k_M = self._batch_unshuffle_ddp(k_M, idx_unshuffle)

        return k_A, k_M

    @paddle.no_grad()
    def _diff_speed(self, im_q, im_k):
        B, C, T, H, W = im_q.shape
        random_indices = paddle.randperm(B) # [B]
        selected_s1 = random_indices[:int(B * self.alpha)] # [B/2]
        selected_s2 = random_indices[int(B * self.alpha):] # [B/2]

        diff_speed = random.choice(self.diff_speed) # [1]
        T_real = T // diff_speed
        speed1 = paddle.arange(0, T, 1)[: T_real]            # speed1 is normal speed [T_r]
        speed2 = paddle.arange(0, T, diff_speed)[: T_real]   # speed2 is randomly selected from self.diff_speed [T / diff_speed]
        im_q_real = paddle.empty([B, C, T_real, H, W]) # [B, C, T_real, H, W]
        im_k_real = paddle.empty_like(im_q_real) # [B, C, T_real, H, W]
        im_k_negative = paddle.empty_like(im_q_real) # [B, C, T_real, H, W]

        tmp = paddle.index_select(im_q, selected_s1, 0)
        im_q_real[selected_s1.numpy()] = paddle.index_select(tmp, speed1, 2) # [B/2, C, T_real, H, W]
        im_q_real[selected_s2.numpy()] = paddle.index_select(paddle.index_select(im_q, selected_s2, 0), speed2, 2) # [B/2, C, T_real, H, W]
        im_k_real[selected_s1.numpy()] = paddle.index_select(paddle.index_select(im_k, selected_s1, 0), speed1, 2) # [B/2, C, T_real, H, W]
        im_k_real[selected_s2.numpy()] = paddle.index_select(paddle.index_select(im_k, selected_s2, 0), speed2, 2) # [B/2, C, T_real, H, W]
        im_k_negative[selected_s1.numpy()] = paddle.index_select(paddle.index_select(im_k, selected_s1, 0), speed2, 2) # [B/2, C, T_real, H, W]
        im_k_negative[selected_s2.numpy()] = paddle.index_select(paddle.index_select(im_k, selected_s2, 0), speed1, 2) # [B/2, C, T_real, H, W]

        k_negative_A, k_negative_M = self._forward_encoder_k(im_k_negative)

        return im_q_real, im_k_real, k_negative_A, k_negative_M


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        reversed_k, speed_k = None, None

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            
            if self.diff_speed is not None:
                im_q, im_k, k_neg_A, k_neg_M = self._diff_speed(im_q, im_k)  # Update im_q, im_k

            k_A, k_M = self._forward_encoder_k(im_k)

        # compute query features
        q_A, q_M = self.encoder_q(im_q)  # queries: NxC

        # compute logits
        # Einstein sum is more intuitive
        # For A-VID task, we consider the clips in the same videos as positive key even they are sampled in different speeds
        # positive logits: Nx1
        # l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        # l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        # l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        l_pos_A1 = paddle.sum(q_A * k_A, axis=1).unsqueeze(-1)
        l_pos_A2 = paddle.sum(q_A * k_neg_A, axis=1).unsqueeze(-1)
        l_pos_M = paddle.sum(q_M * k_M, axis=1).unsqueeze(-1)
        # negative logits: NxK
        # l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])
        # l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)
        l_neg_A = paddle.matmul(q_A, self.queue.clone().detach())
        l_neg_M = paddle.sum(q_M * k_neg_M, axis=1).unsqueeze(-1)
        print(self.queue.clone().detach().shape)

        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T

        # logits: Nx(1+K)
        logits1 = paddle.concat([l_pos_A1, l_neg_A], axis=1)
        logits2 = paddle.concat([l_pos_A2, l_neg_A], axis=1)
        logits_A = (logits1, logits2)
        logits_M = (l_pos_M, l_neg_M)  # l_pos > l_neg_speed


        # labels: positive key indicators
        labels_A = paddle.zeros([logits1.shape[0], 1], dtype='int64')
        labels_M = paddle.ones_like(labels_A)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_neg_A)

        return logits_A, labels_A, logits_M, labels_M



# utils
@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor

    tensors_gather = [paddle.ones_like(tensor)
        for _ in range(paddle.distributed.get_world_size())]
    tensors_gather=[]
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output
