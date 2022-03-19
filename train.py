import torch.nn.functional as F
from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct(part, graph, pos, one_hops)


def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_src(src='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    else:
        raise NotImplementedError


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def run(graph, node_dict, gpb, args):
    
    rank, size = dist.get_rank(), dist.get_world_size()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    boundary = get_boundary(node_dict, gpb)

    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)

    pos = get_pos(node_dict, gpb)
    graph = order_graph(part, graph, gpb, node_dict, pos)
    in_deg = node_dict['in_degree']

    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)

    recv_shape = get_recv_shape(node_dict)

    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline,
                           corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum)

    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)

    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()

    ctx.reducer.init(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, args.n_train))

    best_model, best_acc = None, 0

    if args.grad_corr and args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_grad_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.grad_corr:
        result_file_name = 'results/%s_n%d_p%d_grad.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    elif args.feat_corr:
        result_file_name = 'results/%s_n%d_p%d_feat.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    else:
        result_file_name = 'results/%s_n%d_p%d.txt' % (args.dataset, args.n_partitions, int(args.enable_pipeline))
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur = [], [], []
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        if args.model == 'graphsage':
            logits = model(graph, feat, in_deg)
        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        ctx.buffer.next_epoch()

        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5 and epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)

        if (epoch + 1) % 10 == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        ctx.comm_timer.clear()

        del loss

        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                model_copy, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, result_file_name))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', result_file_name))

    if args.eval and rank == 0:
        if thread is not None:
            model_copy, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        torch.save(best_model.state_dict(), 'model/' + args.graph_name + '_final.pth.tar')
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))
        _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '%d' % args.port
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)
