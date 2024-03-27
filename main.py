from torch import nn
from torch.utils.data import DataLoader
from utils.data_manager import Cell
from utils.data_loader import CellDataset
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from utils.spatial_transforms import *
from models.resnet import *
from models.resnext import *
from train import train
from val import val
from utils.sampler import RandomPidSampler
from utils.losses import TriHardLoss, CenterLoss, CrossEntropyLabelSmooth
from predict import predict
from visualize import compute_feature, display_query_gallery
import os.path as osp
from utils.logger import get_logger
import os
import yaml
import argparse
import warnings
import wandb
import time
from utils.download import download_file, name2id

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MitoReID():
    def __init__(self, config):
        """
            config parameters
            ----------

            model:
                resume_path: str (required)
                    path to the model (pretrain/resume)

                resume: bool (optional, default 0)
                    resume or pretrain
                    1=resume; 0=pretrain

                arch: str (optional, default 'resnet50')
                    model architecture
                    supported architecture: 'resnet18', 'resnet50', 'resnet101', 'resnext101'

                freeze: bool (optional, default 0)
                    1=freeze all layers except the last layer for fine-tune; 0=train all layers

                loss: str (optional, default 'id+trihard')
                    loss function
                    'id'=only use id loss; 'trihard'=only use trihard loss; 'id+trihard'=use id loss and trihard loss

                bnneck: bool (optional, default 1)
                    whether to use bnneck

                downscale_temporal_dimension: bool (optional, default 0)
                    1=downscale temporal dimension over all layers; 0=downscale temporal dimension over the first two layers

                device_ids: str (optional, default '0')
                    device ids
                    ''=use cpu; '0'=use 'cuda:0'; '0,1'=use 'cuda:0' and 'cuda:1'

            input:
                train_path: str (required)
                    path to the train dataset, use ',' to concat different dataset paths

                query_path: str (required)
                    path to the query dataset, use ',' to concat different dataset paths. Set the dataset path to predict when predicting

                gallery_path: str (required)
                    path to the gallery dataset, use ',' to concat different dataset paths

                target_path: str (required)
                    path to the annotation (annotation.xlsx)

                type: str (optional, default 'rgb')
                    data type
                    'rgb'=rgb images; 'flow'=flow images

                id_col: str (optional, default 'moa_id')
                    label column name in annotation.xlsx
                    'moa_id'=classify the drugs by moa id; 'drug_id'=classify the drugs by drug id

                choose_col: str (optional, default 'moa_choose')
                    choosing column name in annotation.xlsx
                    'moa_choose'=choose the moa classes with drugs more than 5 without exception, see 'moa_comment' column for details, 'all'=use all

                num_classes: int (required)
                    num of classes
                    31=MOAs; 1069=drugs

                num_timepoint: int (optional, default 16)
                    num of timepoints

                sample_rate: float (optional, default 1)
                    data sampling rate (between 0 and 1)

                input_size: int (optional, default 112)
                    input image size to the network

                sample_size: int (optional, default 128)
                    resize the raw image due to memery limitation (between 128 and 512, larger than input_size)

                num_instances: int (optional, default 4)
                    number of instances per class for each batch sampling

                augment: bool (optional, default 1)
                    whether to augment the data

                num_workers: int (optional, default 16)
                    Number of threads in data loader

                mean: list (optional, default for type 'rgb' [0.067, 0.102, 0.024], default for type 'flow' [0, 0.498, 0.491])
                    mean value of the dataset

                std: list (optional, default for type 'rgb' [0.042, 0.128, 0.042], default for type 'flow' [1e5, 0.032, 0.035])
                    std value of the dataset

            output:
                model_path: str (required)
                    path to the model to save

                feature_path: str (required)
                    path to the feature to save

                predict_path: str (required)
                    path to the predict result to save

            solver:
                epoch: int (optional, default 100)
                    Number of total epoch to train

                warmup_epoch: int (optional, default 0)
                    'lr_patience' and 'num_mislabel' start to function after the number of 'warmup_epoch'

                lr: float (optional, default 0.001)
                    learning rate

                center_lr: float (optional, default 0.05)
                    learning rate for center loss

                weight: float (optional, default 0.01)
                    weight for center loss

                margin: float (optional, default 0.3)
                    margin for trihard loss

                smooth_epsilon: float (optional, default 0.1)
                    epsilon for label smooth

                num_mislabel: int (optional, default 0)
                    maxmal allowed num of non-control samples predicting as control in a batch, only available when epoch after 'warmup_epoch'

                bs: int (optional, default 128, minimal 64)
                    batch size

                tol: int (optional, default 30)
                    number of epoch to stop the training when no performance is improved

                lr_patience: int (optional, default 3)
                    learning rate patience to decay, only available when scheduler is 'adapt' and epoch after 'warmup_epoch', the number of actual epoch to decay='lr_patience' * 'val_period'

                val_period: int (optional, default 2)
                    interval number of epoch to validate the model

                optimizer: str (optional, default 'sgd')
                    'sgd'=stochastic gradient descent; 'adam'= adam optimizer

                scheduler: str (optional, default 'adapt')
                    'adapt'=ReduceLROnPlateau scheduler; 'cos'=warmup cosine schedular

            """
        with open(config, "r", encoding='utf-8') as f:
            config_json = yaml.load(f, Loader=yaml.SafeLoader)

        self.resume_path = config_json['model']['resume_path']
        self.resume = config_json['model']['resume']
        self.arch = config_json['model']['arch']
        self.freeze = config_json['model']['freeze']
        self.loss = config_json['model']['loss']
        self.bnneck = config_json['model']['bnneck']
        self.downscale_temporal_dimension = config_json['model']['downscale_temporal_dimension']
        self.device_ids = config_json['model']['device_ids']
        self.train_path = config_json['input']['train_path']
        self.query_path = config_json['input']['query_path']
        self.gallery_path = config_json['input']['gallery_path']
        self.target_path = config_json['input']['target_path']
        self.type = config_json['input']['type']
        self.id_col = config_json['input']['id_col']
        self.choose_col = config_json['input']['choose_col']
        self.num_classes = config_json['input']['num_classes']
        self.num_timepoint = config_json['input']['num_timepoint']
        self.sample_rate = config_json['input']['sample_rate']
        self.input_size = config_json['input']['input_size']
        self.sample_size = config_json['input']['sample_size']
        self.num_instances = config_json['input']['num_instances']
        self.augment = config_json['input']['augment']
        self.num_workers = config_json['input']['num_workers']
        self.mean = config_json['input']['mean']
        self.std = config_json['input']['std']
        self.model_path = config_json['output']['model_path']
        self.feature_path = config_json['output']['feature_path']
        self.predict_path = config_json['output']['predict_path']
        self.epoch = config_json['solver']['epoch']
        self.warmup_epoch = config_json['solver']['warmup_epoch']
        self.lr = config_json['solver']['lr']
        self.center_lr = config_json['solver']['center_lr']
        self.weight = config_json['solver']['weight']
        self.margin = config_json['solver']['margin']
        self.smooth_epsilon = config_json['solver']['smooth_epsilon']
        self.num_mislabel = config_json['solver']['num_mislabel']
        self.bs = config_json['solver']['bs']
        self.tol = config_json['solver']['tol']
        self.lr_patience = config_json['solver']['lr_patience']
        self.val_period = config_json['solver']['val_period']
        self.optimizer = config_json['solver']['optimizer']
        self.scheduler = config_json['solver']['scheduler']
        self.config_json = config_json

        if self.model_path != '' and not osp.exists(osp.split(self.model_path)[0]):
            os.makedirs(osp.split(self.model_path)[0])
        if self.feature_path != '' and not osp.exists(osp.split(self.feature_path)[0]):
            os.makedirs(osp.split(self.feature_path)[0])
        if self.predict_path != '' and not osp.exists(osp.split(self.predict_path)[0]):
            os.makedirs(osp.split(self.predict_path)[0])
        self.cmc_path = self.model_path.replace('.pth', '_cmc.csv')
        self.logger = get_logger(self.model_path.replace('.pth', '_log.txt'))

        if not osp.exists(self.resume_path):
            save_folder, name = osp.split(self.resume_path)
            if name in name2id:
                download_file(name, save_folder)

    def load_model(self):
        if self.device_ids != "" and torch.cuda.is_available():
            use_gpu = True
            device_ids = [int(id) for id in self.device_ids.split(",")]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_ids
            self.device = torch.device(f"cuda:{device_ids[0]}")
            self.pin_memory = True
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(1)
        else:
            use_gpu = False
            self.pin_memory = False
            self.device = torch.device("cpu")
            device_ids = None

        if self.type == "flow":
            channel = 2
            self.num_timepoint = self.num_timepoint - 1
        else:
            channel = 3
            self.num_timepoint = self.num_timepoint

        if self.resume:
            pretrain_path = None
        else:
            pretrain_path = self.resume_path
            self.resume_path = None

        if self.arch == 'resnet18':
            self.feat_dim = 512
            model = resnet18(num_classes=self.num_classes, shortcut_type="A",
                             sample_size=self.input_size,
                             channel=channel, downscale_temporal_dimension=self.downscale_temporal_dimension,
                             sample_duration=self.num_timepoint, bnneck=self.bnneck, pretrain=pretrain_path)
        elif self.arch == 'resnet50':
            self.feat_dim = 2048
            model = resnet50(num_classes=self.num_classes, shortcut_type="B",
                             sample_size=self.input_size,
                             channel=channel, downscale_temporal_dimension=self.downscale_temporal_dimension,
                             sample_duration=self.num_timepoint, bnneck=self.bnneck, pretrain=pretrain_path)
        elif self.arch == 'resnet101':
            self.feat_dim = 2048
            model = resnet101(num_classes=self.num_classes, shortcut_type="B",
                              sample_size=self.input_size,
                              channel=channel, downscale_temporal_dimension=self.downscale_temporal_dimension,
                              sample_duration=self.num_timepoint, bnneck=self.bnneck, pretrain=pretrain_path)
        elif self.arch == 'resnext101':
            self.feat_dim = 8192
            model = resnext101(num_classes=self.num_classes, shortcut_type="B",
                               sample_size=self.input_size,
                               channel=channel,
                               downscale_temporal_dimension=self.downscale_temporal_dimension,
                               sample_duration=self.num_timepoint, bnneck=self.bnneck, pretrain=pretrain_path)
        else:
            self.logger.info(f"not supported backbone: {self.arch}")
            return

        if use_gpu:
            model = nn.DataParallel(model, device_ids=device_ids).to(self.device)

        if self.resume_path:
            self.logger.info('resumed model {}'.format(self.resume_path))
            self.checkpoint = torch.load(self.resume_path)
            model.load_state_dict(self.checkpoint['state_dict'], strict=True)
        else:
            self.logger.info('pretrained model {}'.format(pretrain_path))
        return model

    def prepare_data(self, relabel, is_predict=False):
        MEAN, STD = self.mean, self.std
        transform_query = Compose([
            Scale(self.sample_size),
            CenterCrop(self.input_size),
            ToTensor(),
            Normalize(MEAN, STD)
        ])
        transform_gallery = transform_query
        if self.augment:
            transform_train = Compose([
                Scale(self.sample_size),
                RandomRotate(),
                RandomResize(),
                RandomFlip(),
                GaussianBlur(),
                RandomCrop(self.input_size),
                ToTensor(),
                Normalize(MEAN, STD)
            ])
        else:
            transform_train = transform_gallery

        dataset = Cell(self.target_path, self.train_path, self.query_path,
                       self.gallery_path,
                       self.id_col, self.logger, choose_col=self.choose_col, relabel=relabel, is_predict=is_predict,
                       sample_rate=self.sample_rate, model_path=self.model_path)

        if len(dataset.train) > 0:
            if self.loss == 'id':
                sampler = None
                shuffle = True
            else:
                sampler = RandomPidSampler(dataset.train, num_instances=self.num_instances)
                shuffle = False

            train_loader = DataLoader(
                CellDataset(dataset.train, self.num_timepoint, self.type, transform_train),
                batch_size=self.bs, num_workers=self.num_workers,
                pin_memory=self.pin_memory, drop_last=True, shuffle=shuffle, sampler=sampler)
        else:
            train_loader = None
        query_loader = DataLoader(CellDataset(dataset.query, self.num_timepoint, self.type, transform_query),
                                  batch_size=self.bs, num_workers=self.num_workers,
                                  drop_last=False, shuffle=False)
        gallery_loader = DataLoader(
            CellDataset(dataset.gallery, self.num_timepoint, self.type, transform_gallery),
            batch_size=self.bs, num_workers=self.num_workers,
            drop_last=False, shuffle=False)

        return train_loader, query_loader, gallery_loader

    def _train(self):
        assert self.bs >= 64, "bs (batch size) should be larger than 64 for training"

        id = wandb.util.generate_id()
        wandb.init(project='MitoReID', config=self.config_json, resume='allow',
                   name=osp.splitext(osp.basename(self.model_path))[0], id=id)

        model = self.load_model()
        train_loader, query_loader, gallery_loader = self.prepare_data(relabel=True)

        best_metric = 0
        begin_epoch = 0
        if self.loss == 'id':
            criterion_metric = None
        else:
            criterion_metric = TriHardLoss(margin=self.margin)
        if self.loss == 'trihard':
            criterion_class = None
            self.logger.info(f"use {self.loss} loss")
        else:
            criterion_class = CrossEntropyLabelSmooth(num_classes=self.num_classes, device=self.device,
                                                      epsilon=self.smooth_epsilon)
            self.logger.info(f"use {self.loss} loss")
        if criterion_class is not None and criterion_metric is not None:
            criterion_center = CenterLoss(num_classes=self.num_classes, feat_dim=self.feat_dim, device=self.device)
            optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=self.center_lr)
        else:
            criterion_center = None
            optimizer_center = None

        if self.bnneck:
            self.logger.info('use bnneck')
        else:
            self.logger.info('do not use bnneck')
        if self.freeze:
            param = model.module.fc.parameters()
        else:
            param = model.parameters()
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(param, lr=self.lr, weight_decay=5e-4)
        else:  # sgd
            optimizer = torch.optim.SGD(param, lr=self.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
        if self.scheduler == 'adapt':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1,
                                                       patience=self.lr_patience-1,
                                                       verbose=True)
        else:  # cos
            scheduler = self._warmup_cosine_schedular(optimizer)

        if self.resume_path:
            try:
                begin_epoch = self.checkpoint['epoch']
                optimizer.load_state_dict(self.checkpoint['optimizer_state'])
                optimizer.param_groups[0]['lr'] = self.lr
                # scheduler.load_state_dict(self.checkpoint['scheduler_state'])
                if criterion_class is not None and criterion_metric is not None:
                    criterion_center.load_state_dict(self.checkpoint['criterion_center'])
                    optimizer_center.load_state_dict(self.checkpoint['optimizer_center'])
                # best_metric = self.checkpoint['best_metric']
            except Exception as e:
                self.logger.error(f"Exception: {e}!")
                pass

        self.logger.info("start training")
        last_improve = begin_epoch
        epoch = begin_epoch
        save_model = False
        is_best = False
        try:
            while (epoch < self.epoch):
                if epoch > self.warmup_epoch:
                    num_mislabel = self.num_mislabel
                else:
                    num_mislabel = 0
                train_loss, class_loss, metric_loss, center_loss = train(epoch, model, criterion_class,
                                                                         criterion_metric, criterion_center,
                                                                         self.weight, num_mislabel,
                                                                         optimizer, optimizer_center,
                                                                         train_loader, self.device, self.logger)
                log = {'epoch': epoch,
                       'lr': optimizer.param_groups[0]['lr'],
                       'train/loss': train_loss,
                       'train/class_loss': class_loss,
                       'train/metric_loss': metric_loss,
                       'train/center_loss': center_loss}
                wandb.log(log)

                if self.scheduler == 'cos':
                    scheduler.step()
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }
                if criterion_class is not None and criterion_metric is not None:
                    state['criterion_center'] = criterion_center.state_dict()
                    state['optimizer_center'] = optimizer_center.state_dict()

                if (epoch + 1) % self.val_period == 0:
                    rank1, mAP = val('train', model, query_loader, gallery_loader,
                                     criterion_class, self.logger, self.device)
                    metric = mAP
                    if self.scheduler == 'adapt' and epoch > self.warmup_epoch:
                        scheduler.step(metric)
                    log = {'val/mAP': mAP,
                           'val/rank1': rank1}
                    wandb.log(log)
                    is_best = metric > best_metric
                    best_metric = max(metric, best_metric)
                    save_model = True

                if save_model:
                    save_model = False
                    state["best_metric"] = best_metric
                    state["scheduler_state"] = scheduler.state_dict()
                    if is_best:
                        last_improve = epoch
                        self.logger.info("improved")
                        torch.save(state, self.model_path.replace('.pth', '_best.pth'))
                    else:
                        self.logger.info(f"didn't not improve from {best_metric:1%}")
                        torch.save(state, self.model_path)
                    self.logger.info("saved model")

                if epoch < self.warmup_epoch:
                    last_improve = epoch
                if epoch - last_improve > self.tol:
                    self.logger.info("No optimization for a long time, auto-stopping...")
                    break
                epoch += 1
        except Exception as e:
            self.logger.error(e)
            pass

        self.logger.info("finished training")
        best_model_path = self.model_path.replace('.pth', '_best.pth')
        if osp.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            val('test', model, query_loader, gallery_loader, criterion_class, self.logger, self.device,
                cmc_path=self.cmc_path)

    def _warmup_cosine_schedular(self, optimizer, warmup_epoch=10, cosine_epoch=190, n_t=0.5):
        lr_lambda = lambda epoch: (0.9 * epoch / warmup_epoch + 0.1) if epoch < warmup_epoch else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - warmup_epoch) / cosine_epoch)) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - warmup_epoch) / cosine_epoch))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _test(self):
        self.logger.info("test only")
        model = self.load_model()
        train_loader, query_loader, gallery_loader = self.prepare_data(relabel=True)
        if self.loss == 'trihard':
            criterion_class = None
        else:
            criterion_class = CrossEntropyLabelSmooth(num_classes=self.num_classes, device=self.device,
                                                      epsilon=self.smooth_epsilon)
        val('test', model, query_loader, gallery_loader, criterion_class, self.logger,
            self.device, cmc_path=self.cmc_path)

    def _predict(self):
        self.logger.info("predict only")
        model = self.load_model()
        train_loader, query_loader, gallery_loader = self.prepare_data(relabel=False, is_predict=True)
        predict(self.target_path, model, self.predict_path, query_loader, gallery_loader,
                self.device, self.id_col, self.logger)

    def _feature(self):
        self.logger.info("compute features only")
        model = self.load_model()
        train_loader, query_loader, gallery_loader = self.prepare_data(relabel=False)
        compute_feature(model, self.feature_path, query_loader, self.device)


def run(mito, task):
    if task == 'train':
        mito._train()
    elif task == 'test':
        mito._test()
    elif task == 'predict':
        mito._predict()
    elif task == 'feature':
        mito._feature()
    else:
        print('Not supported task! Should be either "train", "test", "predict" or "feature".')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yml', type=str)
    parser.add_argument('--task', default='train', type=str)
    args = parser.parse_args()
    mito = MitoReID(args.config)

    run(mito, args.task)
