import os
import torch
import random
import argparse
import numpy as np
from omegaconf import OmegaConf
import torch.distributed as dist
from SpatialNet import SpatialNet as Model
from utils.loss import LossManager as Loss
from utils.dataloader import SEDataset as Dataset
from utils.dataloader import collate_fn
from utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR
from trainer import Trainer as Trainer

seed = 43
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

def run(rank, config, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12388'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)
    
    shuffle = False if args.world_size > 1 else True
    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
        
    # model = Model(**config['network_config']).to(args.device)
    model = Model(
        dim_input=4,
        dim_output=2,
        num_layers=16,
        encoder_kernel_size=5,
        dim_hidden=128,
        dim_ffn=192,
        num_heads=4,
        dropout=[0, 0, 0],
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=8,
        num_freqs=257,
        full_share=0,
    ).to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler']['kwargs'])
    scheduler = WarmupLR(optimizer, **config['scheduler']['kwargs'])
    
    loss_func = Loss(**config['loss']).to(args.device)

    trainer = Trainer(config=config, model=model,optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='cfg_train.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = OmegaConf.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)