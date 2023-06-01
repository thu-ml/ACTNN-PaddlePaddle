import os
import numpy as np
import paddle.dataset as dataset
import paddle.vision.transforms as transforms
from paddle.io import DataLoader

import paddle
# import paddle.vision.datasets as datasets
from paddle.vision.datasets import DatasetFolder
from paddle.incubate.dataloader import MultiProcessLoader

from actnn import dataloader

from paddle.incubate.hapi.datasets import DatasetBuilder

from paddle.static import InputSpec



DATA_BACKEND_CHOICES = ['paddle']


# try:
#     from nvidia.dali.plugin.pytorch import DALIClassificationIterator
#     from nvidia.dali.pipeline import Pipeline
#     import nvidia.dali.ops as ops
#     import nvidia.dali.types as types
#     DATA_BACKEND_CHOICES.append('dali-gpu')
#     DATA_BACKEND_CHOICES.append('dali-cpu')
# except ImportError:
#     print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
#         if torch.distributed.is_initialized():
#             local_rank = torch.distributed.get_rank()
#             world_size = torch.distributed.get_world_size()
#         else:
#             local_rank = 0
#             world_size = 1
#
#         self.input = ops.FileReader(
#                 file_root = data_dir,
#                 shard_id = local_rank,
#                 num_shards = world_size,
#                 random_shuffle = True)
#         if dali_cpu:
#             dali_device = "cpu"
#             self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
#                                                     random_aspect_ratio=[0.75, 4./3.],
#                                                     random_area=[0.08, 1.0],
#                                                     num_attempts=100)
#         else:
#             dali_device = "gpu"
#             # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
#             # without additional reallocations
#             self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
#                                                       random_aspect_ratio=[0.75, 4./3.],
#                                                       random_area=[0.08, 1.0],
#                                                       num_attempts=100)
#         self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device = "gpu",
#                                             output_dtype = types.FLOAT,
#                                             output_layout = types.NCHW,
#                                             crop = (crop, crop),
#                                             image_type = types.RGB,
#                                             mean = [0.485 * 255,0.456 * 255,0.406 * 255],
#                                             std = [0.229 * 255,0.224 * 255,0.225 * 255])
#         self.coin = ops.CoinFlip(probability = 0.5)
#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name = "Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images.gpu(), mirror = rng)
#         return [output, self.labels]
# class HybridValPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
#         super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
#         if torch.distributed.is_initialized():
#             local_rank = torch.distributed.get_rank()
#             world_size = torch.distributed.get_world_size()
#         else:
#             local_rank = 0
#             world_size = 1
#         self.input = ops.FileReader(
#                 file_root = data_dir,
#                 shard_id = local_rank,
#                 num_shards = world_size,
#                 random_shuffle = False)
#         self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
#         self.res = ops.Resize(device = "gpu", resize_shorter = size)

#         self.cmnp = ops.CropMirrorNormalize(device = "gpu",
#                 output_dtype = types.FLOAT,
#                 output_layout = types.NCHW,
#                 crop = (crop, crop),
#                 image_type = types.RGB,
#                 mean = [0.485 * 255,0.456 * 255,0.406 * 255],
#                 std = [0.229 * 255,0.224 * 255,0.225 * 255])
#     def define_graph(self):
#         self.jpegs, self.labels = self.input(name = "Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images)
#         return [output, self.labels]


class DALIWrapper(object):
    def gen_wrapper(dalipipeline, num_classes, one_hot):
        for data in dalipipeline:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            if one_hot:
                target = paddle.nn.functional.one_hot(target, num_classes)
            yield input, target
        dalipipeline.reset()
    def __init__(self, dalipipeline, num_classes, one_hot):
        self.dalipipeline = dalipipeline
        self.num_classes =  num_classes
        self.one_hot = one_hot
    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline, self.num_classes, self.one_hot)

def get_dali_train_loader(dali_cpu=False):
    def gdtl(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        if paddle.distributed.get_world_size() > 1:
            local_rank = paddle.distributed.get_rank()
            world_size = paddle.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        traindir = os.path.join(data_path, 'train')
        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, crop = 224, dali_cpu=dali_cpu)
        pipe.build()
        train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))
        return DALIWrapper(train_loader, num_classes, one_hot), int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdtl

def get_dali_val_loader():
    def gdvl(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        if paddle.distributed.get_world_size() > 1:
            local_rank = paddle.distributed.get_rank()
            world_size = paddle.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        valdir = os.path.join(data_path, 'val')
        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir,
                crop = 224, size = 256)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))
        return DALIWrapper(val_loader, num_classes, one_hot), int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = paddle.to_tensor([target[1] for target in batch], dtype='int64')
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = paddle.zeros( (len(imgs), 3, h, w), dtype='uint8' )
    for i, img in enumerate(imgs):
        #nump_array = np.asarray(img, dtype=np.uint8)
        nump_array = np.array(np.asarray(img, dtype=np.uint8))
        tens = paddle.to_tensor(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += paddle.to_tensor(nump_array)
    return tensor, targets

def expand(num_classes, dtype, tensor):
    e = paddle.zeros(tensor.shape[0], num_classes, dtype=dtype, device='cuda')
    e = paddle.scatter(e, 1, tensor.unsqueeze(1), 1.0)
    return e



class PrefetchedWrapper(object):
    @staticmethod
    def prefetched_loader(loader, num_classes, fp16, one_hot):
        # if num_classes == 10 or num_classes == 100:   # Cifar10
        #     mean = paddle.to_tensor([0.491 * 255, 0.482 * 255, 0.447 * 255]).cuda().view(1, 3, 1, 1)
        #     std = paddle.to_tensor([0.247 * 255, 0.243 * 255, 0.262 * 255]).cuda().view(1, 3, 1, 1)
        # else:
        mean = paddle.to_tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = paddle.to_tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if fp16:
            mean = mean.half()
            std = std.half()
        stream = paddle.device.cuda.Stream()
        first = True
        for next_indices, next_data in loader:
            next_input, next_target = next_data
            with paddle.device.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, paddle.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, paddle.float, next_target)

                next_input = next_input.sub_(mean).div_(std)
            if not first:
                yield input, target, indices
            else:
                first = False
            paddle.device.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
            N = input.shape[0]
            indices = next_indices.copy()
        yield input, target, indices

    def __init__(self, dataloader, num_classes, fp16, one_hot):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       paddle.io.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.num_classes, self.fp16, self.one_hot)




def get_paddle_train_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = DatasetBuilder().ImageFolder(
            traindir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ]))

    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, worker_init_fn=_worker_init_fn, drop_last=False)

    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)

def get_paddle_val_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = DatasetBuilder().ImageFolder(
            valdir, transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                ]))

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn,
            drop_last=False)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)


def get_paddle_train_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    if num_classes == 10:
        print('Loading CIFAR10')
        train_dataset = datasets.Cifar10(root=data_path, mode='train', transform=transform_train)
    else:
        print('Loading CIFAR100')
        train_dataset = datasets.Cifar100(root=data_path, mode='train', transform=transform_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        collate_fn=fast_collate
    )

    # return train_loader, len(train_loader)
    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)


def get_paddle_val_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    if num_classes == 10:
        val_dataset = datasets.Cifar10(root=data_path, mode='test')
    else:
        val_dataset = datasets.Cifar100(root=data_path, mode='test')

    if paddle.distributed.get_world_size() > 1:
        val_sampler = paddle.io.DistributedBatchSampler(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_sampler = None

    val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=workers, worker_init_fn=_worker_init_fn, collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)


def get_paddle_debug_loader_cifar10(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    if num_classes == 10:
        val_dataset = datasets.Cifar10(root=data_path, mode='test')
    else:
        val_dataset = datasets.Cifar100(root=data_path, mode='test')
    n = val_dataset.data.shape[0]
    n = n//batch_size * batch_size
    val_dataset.data = val_dataset.data[:n]
    val_dataset.targets = val_dataset.targets[:n]

    if paddle.distributed.get_world_size() > 1:
        val_sampler = paddle.io.DistributedBatchSampler(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_sampler = None

    val_loader = paddle.io.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=workers, worker_init_fn=_worker_init_fn, collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)


