import os
from collections import defaultdict

from pytorch_lightning import LightningModule, Trainer
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import dataset_dict
# 位置编码函数, NeRF网络层
from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *
# losses
from utils.losses import *
# 指标
from utils.metrics import *


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.validation_step_outputs = []

        self.hyperparameters = hparams
        # 默认MSE均方损失函数
        self.loss = loss_dict[hparams.loss_type]()
        self.embedding_xyz = Embedding(3, 10)  # xyz默认位置编码维度是4
        self.embedding_dir = Embedding(3, 4)  # 方向默认位置编码维度是4
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        # 定义粗网络
        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            # 定义精细网络
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    # 从batch字典里提取出变量
    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hyperparameters.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hyperparameters.chunk],
                            self.hyperparameters.N_samples,
                            self.hyperparameters.use_disp,
                            self.hyperparameters.perturb,
                            self.hyperparameters.noise_std,
                            self.hyperparameters.N_importance,
                            self.hyperparameters.chunk,  # 块大小在 val 模式下有效
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    # 数据准备阶段
    def setup(self, stage):
        dataset = dataset_dict[self.hyperparameters.dataset_name]
        kwargs = {'root_dir': self.hyperparameters.root_dir, 'img_wh': tuple(self.hyperparameters.img_wh),
                  'spheric_poses': self.hyperparameters.spheric_poses, 'val_num': self.hyperparameters.num_gpus}
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    # 选择要在优化中使用的优化器和学习率调度器
    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hyperparameters, self.models)
        scheduler = get_scheduler(self.hyperparameters, self.optimizer)

        return [self.optimizer], [scheduler]

    # 指定验证数据
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    # 验证步骤 对验证集中的单批数据进行操作
    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        # 进入forward
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # 输出RGB图和深度图
        if batch_nb == 0:
            W, H = self.hyperparameters.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.validation_step_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        # 记录平均损失和psnr
        outputs = self.validation_step_outputs
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        # self.validation_step_outputs.clear()  # 释放内存
        self.log('val/loss', mean_loss, True, sync_dist=True)
        self.log('val/psnr', mean_psnr, True, sync_dist=True)
        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
                }

    # 指定训练数据
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hyperparameters.batch_size,
                          pin_memory=True)

    # 训练步骤 对训练集中的进行按batch处理
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        # 将训练时loss显示在进度条上
        self.log('loss', loss, True)
        self.log('psnr', psnr_, True, sync_dist=True)
        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
                }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}'),
                                          filename='{epoch:d}',
                                          every_n_epochs=hparams.num_epochs,
                                          save_top_k=-1)
    # TQDMProgressBar更新进度条
    callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      accelerator='gpu',
                      logger=logger,
                      devices=hparams.list_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus == 1)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)
