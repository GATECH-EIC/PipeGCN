import torch
from multiprocessing.pool import ThreadPool
from multiprocessing import Event
from helper.timer.timer import *
import queue


class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._pipeline = False
        self._epoch = 0
        self._feat_cpu, self._grad_cpu = [], []
        self._f_buf = []
        self._f_recv, self._b_recv = [], []
        self._f_recv_cpu, self._b_recv_cpu = [], []
        self._f_avg, self._b_avg = [], []
        self._recv_shape = []
        self._pool = None
        self._comm_stream, self._corr_stream = None, None
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        self._corr_momentum = 0
        self._corr_feat, self._corr_grad = False, False
        self._pl, self._pr = [], []

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    def init_buffer(self, num_in, num_all, boundary, f_recv_shape, layer_size, use_pp=False, backend='gloo',
                    pipeline=False, corr_feat=False, corr_grad=False, corr_momentum=0):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._num_in = num_in
        self._boundary = boundary
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._pipeline = pipeline
        self._epoch = 0
        self._recv_shape = f_recv_shape

        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)
                    else:
                        s1 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        tmp1.append(torch.zeros(s1).pin_memory())
                        tmp2.append(torch.zeros(s2).pin_memory())
                        tmp3.append(torch.zeros(s2).pin_memory())
                        tmp4.append(torch.zeros(s1).pin_memory())
                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3
                if i > 0:
                    self._grad_cpu[i] = tmp2
                    self._b_recv_cpu[i] = tmp4

        self._f_buf = [None] * self._n_layers
        self._f_recv, self._b_recv = [], []
        self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []

        self._backend = backend

        self._f_avg, self._b_avg = [None] * self._n_layers, [None] * self._n_layers
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers

        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)
                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([boundary[j].shape[0], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
                    tmp3.append(torch.zeros(s1, device='cuda'))
                    tmp4.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            if i > 0:
                self._b_recv[i] = tmp2
            if corr_feat:
                self._f_avg[i] = tmp3
            if corr_grad and i > 0:
                self._b_avg[i] = tmp4
            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()
        self._corr_momentum = corr_momentum
        self._corr_feat, self._corr_grad = corr_feat, corr_grad
        self._pool = ThreadPool(processes=2*self._n_layers)
        self.__init_pl_pr()

    def next_epoch(self):
        self._epoch += 1

    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                if self._corr_feat:
                    tmp.append(self._f_avg[layer][i])
                else:
                    tmp.append(self._f_recv[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        if self._pipeline is False:
            with comm_timer.timer(f'forward_{layer}'):
                self.__feat_transfer(self._epoch, layer, feat)
                torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]
        else:
            if self._epoch > 0:
                with comm_timer.timer(f'forward_{layer}'):
                    self._f_cpu_event[layer].wait()
                    torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                    self._f_cpu_event[layer].clear()
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, layer, feat))
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]

    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, corr, avg=None, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        self._comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            for i in range(1, size):
                left = (rank - i + size) % size
                right = (rank + i) % size
                r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                req2.put((r2, left))
                if forward:
                    send_cpu[right].copy_(send_gpu[self._boundary[right]])
                else:
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
                r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                req1.append(r1)
            while not req2.empty():
                r, idx = req2.get()
                # TODO: if r.is_completed() run following lines else next r (see issue #30723)
                r.wait()
                recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
                if corr:
                    with torch.cuda.stream(self._corr_stream):
                        self._corr_stream.wait_stream(self._comm_stream)
                        t = avg[idx]
                        t *= self._corr_momentum
                        t += (1 - self._corr_momentum) * recv_gpu[idx]
            # TODO: remove this 'wait'
            for r in req1:
                r.wait()

    def __feat_transfer(self, epoch, layer, feat):
        tag = epoch * 2 * self._n_layers + layer
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                                   tag, self._corr_feat, self._f_avg[layer], forward=True)
            self._f_cuda_event[layer].record(self._comm_stream)
            if self._corr_feat:
                self._f_cuda_event[layer].record(self._corr_stream)
        else:
            raise NotImplementedError
        self._f_cpu_event[layer].set()

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                if self._corr_grad:
                    grad[self._boundary[i]] += self._b_avg[layer][i]
                else:
                    grad[self._boundary[i]] += self._b_recv[layer][i]

    def __grad_hook(self, epoch, layer):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    self.__grad_transfer(epoch, layer, grad)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        self._b_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                        self._b_cpu_event[layer].clear()
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad))
                return grad
        return fn

    def __grad_transfer(self, epoch, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                                   tag, self._corr_grad, self._b_avg[layer], forward=False)
            self._b_cuda_event[layer].record(self._comm_stream)
            if self._corr_grad:
                self._b_cuda_event[layer].record(self._corr_stream)
        else:
            raise NotImplementedError
        self._b_cpu_event[layer].set()
