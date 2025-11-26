import time
import torch
from accelerate import Accelerator
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

from pipeline.model import build_model, load_clip_to_cpu, build_optimizer
from pipeline.dataloader import DataloaderBuilder
from pipeline.criterion import get_loss_function
from peft_clip.model import get_zero_shot_text_features
from util.metrics import Evaluator
from util.util import get_softlabel, process_eval_outputs, test_time_ensemble



class Runner:
    def __init__(self, args, accelerator: Accelerator):
        self.is_main = accelerator.is_main_process
        clip_model = load_clip_to_cpu(args.clip)
        dataloader_builder = DataloaderBuilder(
            name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resolution=clip_model.visual.input_resolution,
            tte=args.tte,
        )
        self.num_classes = dataloader_builder.num_classes
        self.args = args
        self.accelerator = accelerator
        if self.is_main:
            self.writer = SummaryWriter(f"./log/{args.task_name}")
            self.evaluator = Evaluator(
                dataloader_builder.many_shot,
                dataloader_builder.med_shot,
                dataloader_builder.few_shot)
        else:
            self.writer = None
            self.evaluator = None

        model, tuner, head = build_model(args, clip_model, dataloader_builder.num_classes)
        test_loader = dataloader_builder.test()
        if args.hierarchy_prompt:
            # 使用层次信息构筑prompt
            text_features = get_zero_shot_text_features(args.dataset, model, None, accelerator.device)
        else:
            # 使用 a photo of {} 构筑prompt
            text_features = get_zero_shot_text_features(None, model, dataloader_builder.classname, accelerator.device)
        model.init_zero_shot_text_features(text_features)
        if args.model == 'zero-shot':
            assert tuner is None and head is None, "Zero-shot model should not have tuner or head."
        else:
            if args.init_head:
                init_weight = text_features @ model.image_encoder.proj.t()
                init_weight = F.normalize(init_weight, dim=-1)
                model.head.apply_weight(init_weight)
            module_dic = {'model': model, 'tuner': tuner, 'head': head}
            if args.ensemble:
                module_dic.update({'text_learner': model.text_learner})
            optimizer, total_learnable_params = build_optimizer(args.lr, module_dic, args.optim)
            if self.is_main:
                logger.info(f"Total learnable parameters: {total_learnable_params / 1e6:.2f}M")
            train_loader = dataloader_builder.train(args.sampler)
            
            self.optimizer = accelerator.prepare(optimizer)
            self.train_loader = accelerator.prepare(train_loader)
            # 使用accelerate包裹 scheduler会导致余弦退火算法学习率在 epochs / 2 时就变为0
            self.scheduler = CosineAnnealingLR(self.optimizer, args.epochs)

        self.model = accelerator.prepare(model)
        if args.w_mse + args.w_dis > 0:
            softlabel = get_softlabel(args.dataset, args.t_softlabel).to(accelerator.device)
        else:
            softlabel = None
        self.softlabel = softlabel

        criterion = get_loss_function(dataloader_builder, softlabel, args)
        self.test_loader = accelerator.prepare(test_loader)
        self.criterion = accelerator.prepare(criterion)
        self.args = args
        self.accelerator = accelerator
        self.best_acc_train = 0
        self.best_acc_test = 0
        self.save = args.save
        self.task_name = args.task_name

    def __del__(self):
        if self.writer:
            self.writer.close()
        if self.save and self.is_main:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), f'log/{self.args.task_name}/model.pth')  
    def run(self):
        epochs = self.args.epochs
        if self.is_main:
            logger.info(f"Start training {self.args.task_name} with {epochs} epochs")
        for epoch in range(epochs):
            start = time.time()
            self.train(epoch)
            if epoch == 0 or ((epoch + 1) > epochs - 5) or (epoch + 1) % 2 == 0:
                self.eval(epoch)
            self.scheduler.step()
            if self.is_main:
                cur_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('lr', cur_lr, epoch) if self.writer else None
                logger.info(f'Epoch[{epoch + 1}] Time: {time.time() - start:.2f}s, lr: {cur_lr:.6f}')

    def statistics_epoch(self, epoch: int, start_time, loss):
        """call after each epoch to log statistics"""
        if (not self.is_main) or (self.evaluator is None):
            return
        is_training = self.model.training
        accuracy = self.evaluator.evaluate()["accuracy"]
        best_acc_attr = "best_acc_train" if is_training else "best_acc_test"
        best_acc = getattr(self, best_acc_attr)

        if accuracy > best_acc:
            best_acc = accuracy
            setattr(self, best_acc_attr, accuracy)
            if self.save and not is_training:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), f'log/{self.task_name}/best-model.pth')
        elapsed_time = time.time() - start_time
        mark = "Train" if is_training else "Eval"
        num_batch = len(self.train_loader) if is_training else len(self.test_loader)
        avg_loss = loss / num_batch

        logger.info(
            f"Epoch {epoch + 1} {mark}, Time: {elapsed_time:.2f}s, Loss: {avg_loss:.4f}, "
            f"Acc: {accuracy:.2f}% (Best: {best_acc:.2f}%)"
        )

        if self.writer:
            self.writer.add_scalar(f"{mark}/Loss", avg_loss, epoch)
            self.writer.add_scalar(f"{mark}/Accuracy", accuracy, epoch)
        self.evaluator.reset()

    def statistics_batch(self, epoch: int, i: int, loss, outputs, y):
        """call after each batch to log statistics"""
        gather_outputs = self.accelerator.gather(outputs)
        gather_y = self.accelerator.gather(y)
        is_training = self.model.training
        mark = "Train" if is_training else "Eval"
        length = len(self.train_loader) if is_training else len(self.test_loader)
        if self.evaluator:
            self.evaluator.process(gather_outputs, gather_y)
        if (i + 1) % 100 == 0 and self.is_main:
            logger.info(f'Epoch[{epoch + 1}]({mark})[{i + 1}/{length}] Loss: {loss.item():.4f}')

    def train(self, epoch: int):
        start = time.time()
        self.model.train()

        total_loss = 0
        for i, batch in enumerate(self.train_loader):
            x, y, _ = batch
            
            outputs = self.model(x)
            loss = self.criterion(outputs, y.long())
            if isinstance(loss, tuple):
                loss, loss_element = loss
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            total_loss += loss.item()
            if isinstance(outputs, dict):
                outputs = outputs["logit"]
            try:
                if isinstance(loss_element, dict) and self.is_main and (i+1) % 100 == 0:
                    info = ""
                    for k, v in loss_element.items():
                        info += f"{k}: {v:.4f}| "
                    logger.info(f"Loss element: {info}")
            except:
                pass
            self.statistics_batch(epoch, i, loss, outputs, y)
        self.statistics_epoch(epoch, start, total_loss)

    @torch.no_grad()
    def eval(self, epoch=0):
        start = time.time()
        self.model.eval()
        total_loss = 0
        for i, batch in enumerate(self.test_loader):
            x, y, _ = batch

            if self.args.tte:
                outputs = test_time_ensemble(x, self.model)
            else:
                outputs = self.model(x)
            outputs = process_eval_outputs(outputs)
            loss = F.cross_entropy(outputs, y)

            if isinstance(loss, tuple):
                loss, _ = loss
            total_loss += loss.item()
            
            self.statistics_batch(epoch, i, loss, outputs, y)
        self.statistics_epoch(epoch, start, total_loss)
