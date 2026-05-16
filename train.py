# 下面这个代码块导入命令行参数解析工具，方便从 PowerShell 选择 loss、epoch 和 batch size。
import argparse  # argparse 用来读取类似 --loss ce 这样的命令行参数。

# 下面这个代码块导入标准库 csv，方便把每个 epoch 的训练指标写入 CSV 日志文件。
import csv  # csv 是 Python 标准库，不需要额外安装依赖。

# 下面这个代码块导入 Path，方便拼接当前项目内部的数据集路径。
from pathlib import Path  # Path 可以让 Windows 路径拼接更清楚。

# 下面这个代码块导入 PyTorch 主库，训练、验证、设备选择都会用到它。
import torch  # torch 是本项目使用的深度学习框架。

# 下面这个代码块导入 DataLoader，用来把 Dataset 组织成 batch。
from torch.utils.data import DataLoader  # DataLoader 会按 batch 读取图像和 mask。

# 下面这个代码块导入已经完成的数据集模块，只读取当前项目内的数据。
from dataset import IGNORE_INDEX, NUM_CLASSES, StanfordBackgroundDataset  # 这些常量和类来自 src/dataset.py。

# 下面这个代码块导入已经完成的 loss 工厂函数，训练时按名字选择损失函数。
from losses import get_loss_function  # get_loss_function 支持 ce、dice、ce_dice 三种配置。

# 下面这个代码块导入已经完成的指标累计器，验证时统计 pixel accuracy 和 mIoU。
from metrics import SegmentationMetricsTracker  # SegmentationMetricsTracker 可以跨多个 batch 累计指标。

# 下面这个代码块导入已经完成的 U-Net 模型。
from model_unet import UNet  # UNet 是从零实现的语义分割模型。


# 下面这个函数负责解析命令行参数，让短测命令可以控制 loss、epoch 和 batch size。
def parse_args():  # 定义参数解析函数。
    # 下面这个代码块创建参数解析器，并写明这个脚本的用途。
    parser = argparse.ArgumentParser(description="Run a minimal U-Net training loop for HW2 Task 3.")  # 创建命令行参数解析器。
    # 下面这个代码块添加 loss 选择参数。
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default="ce", help="选择损失函数配置。")  # 允许选择三种loss。
    # 下面这个代码块添加 epoch 数参数。
    parser.add_argument("--epochs", type=int, default=1, help="短测运行的 epoch 数。")  # 默认只跑 1 个 epoch。
    # 下面这个代码块添加 batch size 参数。
    parser.add_argument("--batch-size", type=int, default=2, help="每个 batch 的图像数量。")  # 默认 batch size 为 2。
    # 下面这个代码块添加学习率参数。
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam 优化器的学习率。")  # 默认学习率使用 1e-4。
    # 下面这个代码块添加随机种子参数。
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子，方便复现短测结果。")  # 默认使用 42 作为随机种子。
    # 下面这个代码块添加基础通道数参数，短测时用较小模型降低 CPU 压力。
    parser.add_argument("--base-channels", type=int, default=16, help="U-Net 第一层通道数；短测默认用 16。")  # 正式训练时可以改大。
    # 下面这个代码块添加训练 batch 限制，避免短测变成长时间训练。
    parser.add_argument("--max-train-batches", type=int, default=2, help="每个 epoch 最多训练多少个 batch；小于等于 0 表示不限制。")  # 默认只训练 2 个 batch。
    # 下面这个代码块添加验证 batch 限制，避免短测验证完整验证集太慢。
    parser.add_argument("--max-val-batches", type=int, default=2, help="每个 epoch 最多验证多少个 batch；小于等于 0 表示不限制。")  # 默认只验证 2 个 batch。
    # 下面这个代码块添加 checkpoint 输出目录参数。
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="保存最佳模型权重的项目内目录。")  # 默认保存到项目内 checkpoints 目录。
    # 下面这个代码块添加训练日志输出目录参数。
    parser.add_argument("--log-dir", default="outputs/logs", help="保存训练 CSV 日志的项目内目录。")  # 默认保存到项目内 outputs/logs 目录。
    # 下面这个代码块添加外部可视化日志平台参数，默认不启用外部平台。
    parser.add_argument("--logger", choices=["none", "wandb", "swanlab"], default="none", help="选择外部可视化日志平台；none 表示只写本地 CSV。")  # 默认 none 可以保持原训练流程不依赖 wandb 或 swanlab。
    # 下面这个代码块返回解析后的参数。
    return parser.parse_args()  # 返回 argparse.Namespace，后续 main 函数会读取它。


# 下面这个函数根据基础通道数构造 U-Net 四个编码器阶段的通道列表。
def build_features(base_channels):  # 定义通道数构造函数。
    # 下面这个代码块检查基础通道数是否有效。
    if base_channels <= 0:  # 如果通道数小于等于 0，模型无法创建。
        raise ValueError("base_channels 必须大于 0。")  # 给出清楚的报错。
    # 下面这个代码块返回 U-Net 使用的四层通道数。
    return (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)  # 例如 16 会得到 16、32、64、128。


# 下面这个函数负责把传入的输出目录解析为项目内部路径，避免意外写到项目外。
def resolve_project_output_path(project_root, path_text):  # 定义项目内路径解析函数。
    # 下面这个代码块把字符串路径转换为 Path 对象。
    raw_path = Path(path_text)  # raw_path 可能是相对路径，也可能是绝对路径。
    # 下面这个代码块处理相对路径。
    if not raw_path.is_absolute():  # 如果用户传入的是相对路径。
        raw_path = project_root / raw_path  # 就把它解释为相对于项目根目录的路径。
    # 下面这个代码块解析项目根目录的真实绝对路径。
    resolved_project_root = project_root.resolve()  # resolve 可以消除路径中的 .. 等部分。
    # 下面这个代码块解析输出目录的真实绝对路径。
    resolved_path = raw_path.resolve()  # 得到最终准备写入的目录路径。
    # 下面这个代码块检查输出目录是否仍在项目根目录内部。
    if resolved_path != resolved_project_root and resolved_project_root not in resolved_path.parents:  # 如果输出路径不在项目内部。
        raise ValueError(f"输出路径必须位于项目目录内: {resolved_path}")  # 直接报错，避免写到项目外。
    # 下面这个代码块返回安全的项目内路径。
    return resolved_path  # 返回可以安全使用的 Path 对象。


# 下面这个函数负责创建训练输出需要的目录。
def prepare_output_directories(checkpoint_dir, log_dir):  # 定义输出目录准备函数。
    # 下面这个代码块创建 checkpoint 目录。
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 如果目录已存在就直接复用，不会删除任何文件。
    # 下面这个代码块创建日志目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 如果目录已存在就直接复用，不会删除任何文件。


# 下面这个函数负责生成当前 loss 配置对应的最佳 checkpoint 路径。
def build_checkpoint_path(checkpoint_dir, loss_name):  # 定义 checkpoint 路径构造函数。
    # 下面这个代码块构造 checkpoint 文件名。
    checkpoint_name = f"unet_{loss_name}_best.pth"  # 例如 ce 会得到 unet_ce_best.pth。
    # 下面这个代码块返回完整 checkpoint 路径。
    return checkpoint_dir / checkpoint_name  # 返回项目内 checkpoints 目录下的目标文件路径。


# 下面这个函数负责生成当前 loss 配置对应的 CSV 日志路径。
def build_log_path(log_dir, loss_name):  # 定义日志路径构造函数。
    # 下面这个代码块构造日志文件名。
    log_name = f"train_{loss_name}.csv"  # 例如 ce 会得到 train_ce.csv。
    # 下面这个代码块返回完整日志路径。
    return log_dir / log_name  # 返回项目内 outputs/logs 目录下的目标文件路径。


# 下面这个函数负责定义 CSV 日志的列名。
def get_log_fieldnames():  # 定义日志列名函数。
    # 下面这个代码块返回固定顺序的列名列表。
    return ["epoch", "train_loss", "val_loss", "val_acc", "val_miou", "best_val_miou", "checkpoint_saved"]  # 这些字段足够记录本阶段训练情况。


# 下面这个函数负责初始化 CSV 日志文件。
def initialize_log_file(log_path):  # 定义日志初始化函数。
    # 下面这个代码块以写入模式打开日志文件。
    with log_path.open("w", newline="", encoding="utf-8") as file:  # "w" 表示本次训练重新写一个清晰日志。
        # 下面这个代码块创建 CSV 字典写入器。
        writer = csv.DictWriter(file, fieldnames=get_log_fieldnames())  # DictWriter 可以按列名写入字典。
        # 下面这个代码块写入表头。
        writer.writeheader()  # CSV 第一行记录列名。


# 下面这个函数负责把一个 epoch 的指标追加写入 CSV 日志。
def append_log_row(log_path, log_row):  # 定义日志追加函数。
    # 下面这个代码块以追加模式打开日志文件。
    with log_path.open("a", newline="", encoding="utf-8") as file:  # "a" 表示在已有表头后追加一行。
        # 下面这个代码块创建 CSV 字典写入器。
        writer = csv.DictWriter(file, fieldnames=get_log_fieldnames())  # 使用和表头一致的列顺序。
        # 下面这个代码块写入当前 epoch 的日志行。
        writer.writerow(log_row)  # 把一个 epoch 的指标写入 CSV。


# 下面这个函数负责把当前 epoch 的指标整理成 CSV 日志中的一行。
def build_log_row(epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved):  # 定义日志行构造函数。
    # 下面这个代码块返回一行日志字典。
    return {  # 字典的键必须和 get_log_fieldnames() 中的列名一致。
        "epoch": epoch,  # 记录当前 epoch 编号。
        "train_loss": f"{train_loss:.6f}",  # 记录训练集平均 loss。
        "val_loss": f"{val_metrics['val_loss']:.6f}",  # 记录验证集平均 loss。
        "val_acc": f"{val_metrics['pixel_accuracy']:.6f}",  # 记录验证集 pixel accuracy。
        "val_miou": f"{val_metrics['miou']:.6f}",  # 记录验证集 mIoU。
        "best_val_miou": f"{best_val_miou:.6f}",  # 记录截至当前 epoch 的最佳 mIoU。
        "checkpoint_saved": str(checkpoint_saved),  # 记录本 epoch 是否保存了新的最佳 checkpoint。
    }  # 日志行字典构造结束。


# 下面这个函数负责把当前 epoch 的指标整理成外部日志平台需要的数字字典。
def build_external_log_metrics(epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved):  # 定义外部日志指标构造函数。
    # 下面这个代码块返回 wandb 或 swanlab 可以直接记录的指标字典。
    return {  # 外部平台更适合记录数字值，而不是 CSV 中的字符串值。
        "epoch": epoch,  # 记录当前 epoch 编号。
        "train_loss": train_loss,  # 记录训练集平均 loss。
        "val_loss": val_metrics["val_loss"],  # 记录验证集平均 loss。
        "val_acc": val_metrics["pixel_accuracy"],  # 记录验证集 pixel accuracy。
        "val_miou": val_metrics["miou"],  # 记录验证集 mIoU。
        "best_val_miou": best_val_miou,  # 记录截至当前 epoch 的最佳 mIoU。
        "checkpoint_saved": int(checkpoint_saved),  # 用 0 或 1 记录本 epoch 是否保存了新的最佳 checkpoint。
    }  # 外部日志指标字典构造结束。


# 下面这个类负责把训练指标可选地发送到 wandb 或 swanlab。
class TrainingLogger:  # 定义一个轻量日志记录器，避免把外部平台逻辑散落在训练循环里。
    # 下面这个方法负责初始化日志记录器。
    def __init__(self, logger_name, args):  # logger_name 来自 --logger 参数，args 保存本次训练配置。
        # 下面这个代码块保存选择的日志平台名称。
        self.logger_name = logger_name  # 可能的取值是 none、wandb、swanlab。
        # 下面这个代码块准备保存外部平台模块对象。
        self.module = None  # none 模式下不会导入任何外部平台模块。
        # 下面这个代码块准备保存外部平台 run 对象。
        self.run = None  # 有些平台会从 init 返回一个 run 对象。
        # 下面这个代码块处理默认 none 模式。
        if self.logger_name == "none":  # none 表示不启用 wandb 或 swanlab。
            print("external logger: none")  # 明确说明当前只使用本地 CSV 日志。
            return  # none 模式不需要继续初始化外部平台。
        # 下面这个代码块为外部平台构造一个简单的实验名。
        experiment_name = f"unet_{args.loss}"  # 例如 ce 会得到 unet_ce。
        # 下面这个代码块把 argparse 参数转换为普通字典，方便外部平台保存超参数。
        config = vars(args).copy()  # copy 可以避免后续意外修改 args 本身。
        # 下面这个代码块根据选择初始化 wandb。
        if self.logger_name == "wandb":  # 显式选择 wandb 时才导入 wandb。
            self.module = self._import_logger_module("wandb")  # 动态导入 wandb，避免 none 模式依赖它。
            self.run = self.module.init(project="hw2-unet", name=experiment_name, config=config)  # 创建 wandb 实验并记录配置。
            print("external logger: wandb")  # 打印当前启用的外部日志平台。
        # 下面这个代码块根据用户选择初始化 swanlab。
        elif self.logger_name == "swanlab":  # 显式选择 swanlab 时才导入 swanlab。
            self.module = self._import_logger_module("swanlab")  # 动态导入 swanlab，避免 none 模式依赖它。
            self.run = self.module.init(project="hw2-unet", experiment_name=experiment_name, config=config)  # 创建 swanlab 实验并记录配置。
            print("external logger: swanlab")  # 打印当前启用的外部日志平台。

    # 下面这个静态方法负责导入外部日志平台模块，并在缺少依赖时给出清楚错误。
    @staticmethod  # 这个方法不依赖某个具体对象实例。
    def _import_logger_module(module_name):  # module_name 是准备导入的模块名。
        # 下面这个代码块尝试导入外部日志平台模块。
        try:  # 用 try 捕获模块不存在的情况。
            return __import__(module_name)  # 动态导入 wandb 或 swanlab。
        # 下面这个代码块处理没有安装外部日志平台的情况。
        except ImportError as error:  # 如果环境里没有安装对应模块，就会进入这里。
            raise RuntimeError(f"你选择了 --logger {module_name}，但当前环境没有安装 {module_name}。请先安装或改用 --logger none。") from error  # 报错信息说明如何处理。

    # 下面这个方法负责记录一个 epoch 的训练和验证指标。
    def log_epoch(self, epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved):  # 接收当前 epoch 的核心指标。
        # 下面这个代码块在 none 模式下直接跳过外部日志记录。
        if self.logger_name == "none":  # none 模式只写 CSV，不访问外部平台。
            return  # 不做任何外部日志操作。
        # 下面这个代码块构造要发送给外部平台的指标。
        metrics = build_external_log_metrics(epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved)  # 得到数字指标字典。
        # 下面这个代码块检查外部平台是否提供 log 函数。
        if not hasattr(self.module, "log"):  # wandb 和 swanlab 通常都提供模块级 log 函数。
            raise RuntimeError(f"{self.logger_name} 模块没有找到 log 函数，无法记录训练指标。")  # 给出清楚错误，避免静默失败。
        # 下面这个代码块把指标发送到外部日志平台。
        self.module.log(metrics, step=epoch)  # 使用 epoch 作为横轴 step，便于画训练曲线。

    # 下面这个方法负责在训练结束后关闭外部日志平台。
    def finish(self):  # 定义日志结束函数。
        # 下面这个代码块在 none 模式下直接返回。
        if self.logger_name == "none":  # none 模式没有外部 run 需要关闭。
            return  # 不做任何结束操作。
        # 下面这个代码块优先调用模块级 finish 函数。
        if hasattr(self.module, "finish"):  # wandb 和 swanlab 常见用法都支持 finish。
            self.module.finish()  # 正常结束外部日志记录。
            return  # 已经结束后直接返回。
        # 下面这个代码块兼容某些平台把 finish 放在 run 对象上的情况。
        if self.run is not None and hasattr(self.run, "finish"):  # 如果 run 对象存在并支持 finish。
            self.run.finish()  # 调用 run 对象的结束方法。


# 下面这个函数负责把当前最佳模型保存为 checkpoint。
def save_checkpoint(checkpoint_path, epoch, model, optimizer, loss_name, features, args, train_loss, val_metrics, best_val_miou):  # 定义 checkpoint 保存函数。
    # 下面这个代码块整理 checkpoint 中需要保存的信息。
    checkpoint = {  # checkpoint 是一个普通字典，后续 evaluate.py 可以读取它。
        "epoch": epoch,  # 保存当前 epoch 编号。
        "model_state_dict": model.state_dict(),  # 保存模型参数。
        "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态，方便未来继续训练。
        "loss_name": loss_name,  # 保存当前使用的 loss 配置。
        "num_classes": NUM_CLASSES,  # 保存类别数，方便加载模型时检查一致性。
        "ignore_index": IGNORE_INDEX,  # 保存忽略标签，方便后续评估时复用。
        "features": features,  # 保存 U-Net 通道配置，方便重建同结构模型。
        "base_channels": args.base_channels,  # 保存基础通道数。
        "batch_size": args.batch_size,  # 保存训练时的 batch size。
        "learning_rate": args.learning_rate,  # 保存训练时的学习率。
        "train_loss": train_loss,  # 保存当前 epoch 的训练 loss。
        "val_loss": val_metrics["val_loss"],  # 保存当前 epoch 的验证 loss。
        "val_acc": val_metrics["pixel_accuracy"],  # 保存当前 epoch 的验证 pixel accuracy。
        "val_miou": val_metrics["miou"],  # 保存当前 epoch 的验证 mIoU。
        "best_val_miou": best_val_miou,  # 保存截至当前 epoch 的最佳 mIoU。
    }  # checkpoint 字典构造结束。
    # 下面这个代码块把 checkpoint 写入磁盘。
    torch.save(checkpoint, checkpoint_path)  # 保存为 .pth 文件，供后续 evaluate.py 或正式报告使用。


# 下面这个函数负责创建训练集和验证集的 DataLoader。
def build_dataloaders(dataset_root, batch_size, seed):  # 定义 DataLoader 构造函数。
    # 下面这个代码块创建训练集 Dataset。
    train_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="train", seed=seed)  # 训练集使用固定 seed 的 80% 数据。
    # 下面这个代码块创建验证集 Dataset。
    val_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="val", seed=seed)  # 验证集使用固定 seed 的 10% 数据。
    # 下面这个代码块创建训练 DataLoader。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=segmentation_collate_fn)  # 使用自定义 collate_fn 处理不同尺寸图像。
    # 下面这个代码块创建验证 DataLoader。
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=segmentation_collate_fn)  # 验证阶段同样需要处理不同尺寸图像。
    # 下面这个代码块返回两个 DataLoader。
    return train_loader, val_loader  # 返回训练和验证数据加载器。


# 下面这个函数负责把一张图像和对应 mask padding 到指定的高和宽。
def pad_image_and_mask(image, mask, target_height, target_width):  # 定义单样本 padding 函数。
    # 下面这个代码块读取图像原始通道数、高度和宽度。
    channels, height, width = image.shape  # image 的形状是 [3, H, W]。
    # 下面这个代码块创建 padding 后的图像 tensor，默认填充值为 0.0。
    padded_image = torch.zeros((channels, target_height, target_width), dtype=image.dtype)  # padding 图像区域使用黑色像素 0.0。
    # 下面这个代码块创建 padding 后的 mask tensor，默认填充值为 IGNORE_INDEX。
    padded_mask = torch.full((target_height, target_width), IGNORE_INDEX, dtype=mask.dtype)  # padding 标签区域在 loss 和 metrics 中会被忽略。
    # 下面这个代码块把原始图像复制到 padding 图像左上角。
    padded_image[:, :height, :width] = image  # 保留原始图像内容，不改变原始像素值。
    # 下面这个代码块把原始 mask 复制到 padding mask 左上角。
    padded_mask[:height, :width] = mask  # 保留原始标签内容，不改变原始类别编号。
    # 下面这个代码块返回 padding 后的一对 tensor。
    return padded_image, padded_mask  # 返回形状分别为 [3, target_height, target_width] 和 [target_height, target_width] 的 tensor。


# 下面这个函数负责把一个 batch 中不同尺寸的样本整理成可以训练的统一 tensor。
def segmentation_collate_fn(batch):  # 定义语义分割任务专用的 batch 拼接函数。
    # 下面这个代码块检查 batch 是否为空。
    if len(batch) == 0:  # 正常 DataLoader 不会传入空 batch，这里只是提前保护。
        raise RuntimeError("DataLoader 收到了空 batch。")  # 如果真的发生空 batch，就给出清楚报错。
    # 下面这个代码块分别取出 batch 里的所有图像和 mask。
    images, masks = zip(*batch)  # images 是若干个 [3, H, W] tensor，masks 是若干个 [H, W] tensor。
    # 下面这个代码块计算当前 batch 中最大的图像高度。
    max_height = max(image.shape[1] for image in images)  # 不同图片高度可能不同，所以取最大值作为 padding 目标。
    # 下面这个代码块计算当前 batch 中最大的图像宽度。
    max_width = max(image.shape[2] for image in images)  # 不同图片宽度可能不同，所以取最大值作为 padding 目标。
    # 下面这个代码块创建列表，用来保存 padding 后的图像。
    padded_images = []  # 后面会把每张 padding 后的图像放进这个列表。
    # 下面这个代码块创建列表，用来保存 padding 后的 mask。
    padded_masks = []  # 后面会把每张 padding 后的 mask 放进这个列表。
    # 下面这个代码块逐个处理 batch 中的样本。
    for image, mask in zip(images, masks):  # 每次同时取出一张图像和它对应的 mask。
        # 下面这个代码块把当前样本 padding 到当前 batch 的最大高宽。
        padded_image, padded_mask = pad_image_and_mask(image, mask, max_height, max_width)  # 得到统一尺寸的 image 和 mask。
        # 下面这个代码块保存 padding 后的图像。
        padded_images.append(padded_image)  # 加入图像列表，等待后面 stack。
        # 下面这个代码块保存 padding 后的 mask。
        padded_masks.append(padded_mask)  # 加入 mask 列表，等待后面 stack。
    # 下面这个代码块把图像列表堆叠成一个 batch tensor。
    image_batch = torch.stack(padded_images, dim=0)  # 得到形状 [B, 3, max_height, max_width]。
    # 下面这个代码块把 mask 列表堆叠成一个 batch tensor。
    mask_batch = torch.stack(padded_masks, dim=0)  # 得到形状 [B, max_height, max_width]。
    # 下面这个代码块返回 DataLoader 需要的 batch。
    return image_batch, mask_batch  # 返回可以直接输入模型和 loss 的 batch。


# 下面这个函数负责把一个 batch 的图像和 mask 移动到 CPU 或 GPU。
def move_batch_to_device(images, masks, device):  # 定义 batch 设备移动函数。
    # 下面这个代码块移动图像 tensor。
    images = images.to(device)  # images 形状是 [B, 3, H, W]，类型是 float32。
    # 下面这个代码块移动 mask tensor。
    masks = masks.to(device)  # masks 形状是 [B, H, W]，类型是 long。
    # 下面这个代码块返回移动后的 batch。
    return images, masks  # 返回同一设备上的 images 和 masks。


# 下面这个函数负责判断当前 epoch 是否应该提前停止 batch 循环。
def reached_batch_limit(batch_index, max_batches):  # 定义 batch 限制判断函数。
    # 下面这个代码块处理不限制 batch 数的情况。
    if max_batches <= 0:  # 小于等于 0 表示跑完整个 DataLoader。
        return False  # 不提前停止。
    # 下面这个代码块判断是否已经达到短测 batch 上限。
    return batch_index >= max_batches  # batch_index 从 1 开始，所以达到上限就停止。


# 下面这个函数负责训练一个 epoch，但可以用 max_batches 限制只跑少量 batch。
def train_one_epoch(model, train_loader, loss_fn, optimizer, device, max_batches):  # 定义单个 epoch 的训练函数。
    # 下面这个代码块把模型切换到训练模式。
    model.train()  # 训练模式会启用 BatchNorm 的训练行为。
    # 下面这个代码块初始化 loss 累计值。
    total_loss = 0.0  # 用来累计每个 batch 的 loss。
    # 下面这个代码块初始化已经处理的 batch 数。
    total_batches = 0  # 用来计算平均训练 loss。
    # 下面这个代码块逐个读取训练 batch。
    for batch_index, (images, masks) in enumerate(train_loader, start=1):  # batch_index 从 1 开始更适合打印。
        # 下面这个代码块把当前 batch 移动到目标设备。
        images, masks = move_batch_to_device(images, masks, device)  # 保证模型输入和标签在同一个设备上。
        # 下面这个代码块清空上一轮反向传播留下的梯度。
        optimizer.zero_grad()  # 每个 batch 反向传播前都要清空梯度。
        # 下面这个代码块执行 U-Net 前向传播。
        logits = model(images)  # logits 形状应为 [B, NUM_CLASSES, H, W]。
        # 下面这个代码块计算当前 batch 的损失。
        loss = loss_fn(logits, masks)  # masks 形状是 [B, H, W]。
        # 下面这个代码块进行反向传播。
        loss.backward()  # 根据 loss 计算模型参数的梯度。
        # 下面这个代码块更新模型参数。
        optimizer.step()  # Adam 根据梯度更新 U-Net 参数。
        # 下面这个代码块累计 loss。
        total_loss += loss.item()  # loss.item() 把标量 tensor 转成 Python 数字。
        # 下面这个代码块累计 batch 数。
        total_batches += 1  # 记录已经完成了一个训练 batch。
        # 下面这个代码块打印当前训练 batch 的简短日志。
        print(f"train batch {batch_index}: loss={loss.item():.6f}")  # 打印当前 batch 的训练 loss。
        # 下面这个代码块在短测达到 batch 上限时提前停止。
        if reached_batch_limit(batch_index, max_batches):  # 判断是否达到 max_train_batches。
            break  # 只做短测，不继续跑完整训练集。
    # 下面这个代码块检查是否真的训练过至少一个 batch。
    if total_batches == 0:  # 如果一个 batch 都没有跑，说明配置有问题。
        raise RuntimeError("训练阶段没有处理任何 batch。")  # 给出清楚的错误。
    # 下面这个代码块返回平均训练 loss。
    return total_loss / total_batches  # 返回本 epoch 的平均训练损失。


# 下面这个函数负责验证一个 epoch，但可以用 max_batches 限制只跑少量 batch。
def validate_one_epoch(model, val_loader, loss_fn, device, max_batches):  # 定义单个 epoch 的验证函数。
    # 下面这个代码块把模型切换到验证模式。
    model.eval()  # 验证模式会固定 BatchNorm 的行为。
    # 下面这个代码块初始化 loss 累计值。
    total_loss = 0.0  # 用来累计验证 loss。
    # 下面这个代码块初始化已经处理的 batch 数。
    total_batches = 0  # 用来计算平均验证 loss。
    # 下面这个代码块创建指标累计器。
    tracker = SegmentationMetricsTracker(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)  # 用于累计 pixel accuracy 和 mIoU。
    # 下面这个代码块关闭梯度记录，节省验证阶段的显存和时间。
    with torch.no_grad():  # 验证阶段不需要反向传播。
        # 下面这个代码块逐个读取验证 batch。
        for batch_index, (images, masks) in enumerate(val_loader, start=1):  # batch_index 从 1 开始更适合打印。
            # 下面这个代码块把当前 batch 移动到目标设备。
            images, masks = move_batch_to_device(images, masks, device)  # 保证模型输入和标签在同一个设备上。
            # 下面这个代码块执行 U-Net 前向传播。
            logits = model(images)  # logits 形状应为 [B, NUM_CLASSES, H, W]。
            # 下面这个代码块计算当前 batch 的验证损失。
            loss = loss_fn(logits, masks)  # 验证 loss 只计算，不反向传播。
            # 下面这个代码块累计验证 loss。
            total_loss += loss.item()  # 把当前 batch 的 loss 加到总和里。
            # 下面这个代码块累计 batch 数。
            total_batches += 1  # 记录已经完成了一个验证 batch。
            # 下面这个代码块更新验证指标。
            tracker.update(logits, masks)  # 用 logits 和真实 mask 统计 accuracy 与 mIoU。
            # 下面这个代码块打印当前验证 batch 的简短日志。
            print(f"val batch {batch_index}: loss={loss.item():.6f}")  # 打印当前 batch 的验证 loss。
            # 下面这个代码块在短测达到 batch 上限时提前停止。
            if reached_batch_limit(batch_index, max_batches):  # 判断是否达到 max_val_batches。
                break  # 只做短测，不继续跑完整验证集。
    # 下面这个代码块检查是否真的验证过至少一个 batch。
    if total_batches == 0:  # 如果一个 batch 都没有跑，说明配置有问题。
        raise RuntimeError("验证阶段没有处理任何 batch。")  # 给出清楚的错误。
    # 下面这个代码块计算平均验证 loss。
    val_loss = total_loss / total_batches  # 得到本 epoch 的平均验证损失。
    # 下面这个代码块计算累计指标。
    metrics = tracker.compute()  # 得到 pixel_accuracy 和 miou。
    # 下面这个代码块把验证 loss 放进指标字典。
    metrics["val_loss"] = val_loss  # 方便 main 函数统一打印。
    # 下面这个代码块返回验证结果。
    return metrics  # 返回包含 val_loss、pixel_accuracy、miou 的字典。


# 下面这个函数是训练脚本主入口，负责把数据、模型、loss、优化器和循环串起来。
def main():  # 定义主函数。
    # 下面这个代码块解析命令行参数。
    args = parse_args()  # 读取在 PowerShell 中传入的参数。
    # 下面这个代码块固定 PyTorch 随机种子。
    torch.manual_seed(args.seed)  # 让模型初始化和 DataLoader 打乱尽量可复现。
    # 下面这个代码块找到项目根目录。
    project_root = Path(__file__).resolve().parents[1]  # train.py 位于 src 下，所以 parents[1] 是项目根目录。
    # 下面这个代码块拼接 Stanford Background Dataset 的项目内路径。
    dataset_root = project_root / "data" / "stanford_background" / "iccv09Data"  # 数据集主体目录。
    # 下面这个代码块解析 checkpoint 输出目录。
    checkpoint_dir = resolve_project_output_path(project_root, args.checkpoint_dir)  # 保证 checkpoint 写在项目目录内部。
    # 下面这个代码块解析日志输出目录。
    log_dir = resolve_project_output_path(project_root, args.log_dir)  # 保证 CSV 日志写在项目目录内部。
    # 下面这个代码块创建输出目录。
    prepare_output_directories(checkpoint_dir, log_dir)  # 如果目录不存在，就创建项目内目录。
    # 下面这个代码块生成当前 loss 对应的 checkpoint 路径。
    checkpoint_path = build_checkpoint_path(checkpoint_dir, args.loss)  # 例如 checkpoints/unet_ce_best.pth。
    # 下面这个代码块生成当前 loss 对应的日志路径。
    log_path = build_log_path(log_dir, args.loss)  # 例如 outputs/logs/train_ce.csv。
    # 下面这个代码块初始化本次训练的 CSV 日志文件。
    initialize_log_file(log_path)  # 写入 CSV 表头，方便每个 epoch 追加指标。
    # 下面这个代码块选择训练设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 就用 GPU，否则用 CPU。
    # 下面这个代码块构造 DataLoader。
    train_loader, val_loader = build_dataloaders(dataset_root, args.batch_size, args.seed)  # 创建训练和验证加载器。
    # 下面这个代码块构造短测用的 U-Net 通道配置。
    features = build_features(args.base_channels)  # 例如 base_channels=16 时得到 (16, 32, 64, 128)。
    # 下面这个代码块创建 U-Net 模型并移动到目标设备。
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, features=features).to(device)  # 模型输入是 RGB，输出是 8 类 logits。
    # 下面这个代码块创建损失函数并移动到目标设备。
    loss_fn = get_loss_function(args.loss, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)  # 按参数选择 ce、dice 或 ce_dice。
    # 下面这个代码块创建 Adam 优化器。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # Adam 是常用的深度学习优化器。
    # 下面这个代码块创建外部日志记录器；--logger none 时不会导入外部平台。
    external_logger = TrainingLogger(args.logger, args)  # 负责可选地把 epoch 指标记录到 wandb 或 swanlab。
    # 下面这个代码块打印本次短测配置。
    print(f"device: {device}")  # 打印当前使用 CPU 还是 GPU。
    # 下面这个代码块打印数据集路径。
    print(f"dataset_root: {dataset_root}")  # 打印数据集读取位置，便于检查路径。
    # 下面这个代码块打印数据集大小。
    print(f"train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")  # 打印训练集和验证集样本数。
    # 下面这个代码块打印核心超参数。
    print(f"loss: {args.loss}, epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.learning_rate}")  # 打印训练配置。
    # 下面这个代码块打印模型通道配置和短测 batch 限制。
    print(f"features: {features}, max_train_batches: {args.max_train_batches}, max_val_batches: {args.max_val_batches}")  # 打印短测限制。
    # 下面这个代码块打印外部日志平台配置。
    print(f"logger: {args.logger}")  # 打印当前是否启用 wandb 或 swanlab。
    # 下面这个代码块打印 checkpoint 保存路径。
    print(f"checkpoint_path: {checkpoint_path}")  # 打印最佳模型权重的保存位置。
    # 下面这个代码块打印 CSV 日志路径。
    print(f"log_path: {log_path}")  # 打印训练日志的保存位置。
    # 下面这个代码块初始化最佳 mIoU。
    best_val_miou = -1.0  # 第一轮验证后一定会保存一次 checkpoint。
    # 下面这个代码块按 epoch 执行训练和验证。
    for epoch in range(1, args.epochs + 1):  # epoch 从 1 开始，便于阅读日志。
        # 下面这个代码块打印 epoch 开始信息。
        print(f"epoch {epoch}/{args.epochs} start")  # 提示当前 epoch 开始。
        # 下面这个代码块执行一个训练 epoch。
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.max_train_batches)  # 返回平均训练 loss。
        # 下面这个代码块执行一个验证 epoch。
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, args.max_val_batches)  # 返回验证 loss 和指标。
        # 下面这个代码块判断当前 epoch 是否得到新的最佳 mIoU。
        checkpoint_saved = val_metrics["miou"] > best_val_miou  # 如果当前 mIoU 更高，就保存 checkpoint。
        # 下面这个代码块在当前模型更好时更新最佳 mIoU 并保存权重。
        if checkpoint_saved:  # 第一轮或更高 mIoU 时进入这里。
            best_val_miou = val_metrics["miou"]  # 更新截至目前的最佳验证 mIoU。
            save_checkpoint(checkpoint_path, epoch, model, optimizer, args.loss, features, args, train_loss, val_metrics, best_val_miou)  # 保存最佳模型权重。
            print(f"saved best checkpoint: {checkpoint_path}")  # 打印 checkpoint 保存信息。
        # 下面这个代码块构造当前 epoch 的 CSV 日志行。
        log_row = build_log_row(epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved)  # 把当前 epoch 指标整理为字典。
        # 下面这个代码块把当前 epoch 的指标追加到日志文件。
        append_log_row(log_path, log_row)  # 每个 epoch 结束后立刻写日志，避免中途停止时丢失记录。
        # 下面这个代码块把当前 epoch 的指标可选地发送到 wandb 或 swanlab。
        external_logger.log_epoch(epoch, train_loss, val_metrics, best_val_miou, checkpoint_saved)  # none 模式会直接跳过，wandb/swanlab 模式会记录曲线。
        # 下面这个代码块打印日志更新信息。
        print(f"updated log: {log_path}")  # 提示当前 epoch 的日志已经写入 CSV。
        # 下面这个代码块打印 epoch 汇总结果。
        print(f"epoch {epoch}/{args.epochs} summary: train_loss={train_loss:.6f}, val_loss={val_metrics['val_loss']:.6f}, val_acc={val_metrics['pixel_accuracy']:.6f}, val_miou={val_metrics['miou']:.6f}, best_val_miou={best_val_miou:.6f}")  # 打印本轮结果。
    # 下面这个代码块打印短测完成信息。
    print("minimal training loop finished.")  # 表示最小训练循环已经正常结束。
    # 下面这个代码块结束外部日志记录。
    external_logger.finish()  # none 模式不会做任何事，wandb/swanlab 模式会正常结束 run。


# 下面这个代码块保证只有直接运行 python src/train.py 时才执行训练。
if __name__ == "__main__":  # 判断当前文件是否作为脚本直接运行。
    main()  # 调用主函数，启动最小训练短测。
