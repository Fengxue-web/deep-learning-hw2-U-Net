# 本文件实现 U-Net checkpoint 的最小评估脚本，评估阶段不训练模型。

# 下面这个代码块导入命令行参数解析工具，用来读取 --checkpoint、--split 等参数。
import argparse  # argparse 是 Python 标准库，用来解析 PowerShell 中传入的命令行参数。

# 下面这个代码块导入 Path，用来安全地拼接和检查当前项目内部的文件路径。
from pathlib import Path  # Path 可以让 Windows 路径拼接更清晰，也方便检查文件是否存在。

# 下面这个代码块导入 PyTorch，评估时需要用它加载 checkpoint、选择设备、关闭梯度。
import torch  # torch 是本项目使用的深度学习框架。

# 下面这个代码块导入 DataLoader，用来把 Dataset 按 batch 组织起来。
from torch.utils.data import DataLoader  # DataLoader 会逐批读取图像 tensor 和 mask tensor。

# 下面这个代码块导入已经实现的数据集类，用来读取 val 或 test split。
from dataset import StanfordBackgroundDataset  # StanfordBackgroundDataset 会返回 image 和 mask。

# 下面这个代码块导入已经实现的 loss 工厂函数，用 checkpoint 中的 loss_name 重建 loss。
from losses import get_loss_function  # get_loss_function 支持 ce、dice、ce_dice 三种配置。

# 下面这个代码块导入已经实现的指标累计器，用来统计 pixel accuracy 和 mIoU。
from metrics import SegmentationMetricsTracker  # SegmentationMetricsTracker 可以跨多个 batch 累计指标。

# 下面这个代码块导入已经实现的 U-Net 模型类，用 checkpoint 中的结构参数重建模型。
from model_unet import UNet  # UNet 是从零实现的语义分割模型。

# 下面这个代码块复用 train.py 中已经写好的 batch 处理工具，确保评估和训练使用同样的 padding 规则。
from train import move_batch_to_device, reached_batch_limit, segmentation_collate_fn  # 这些函数只被调用，不会修改 train.py。


# 下面这个函数负责解析评估脚本的命令行参数。
def parse_args():  # 定义命令行参数解析函数。
    # 下面这个代码块创建参数解析器，并说明这个脚本的用途。
    parser = argparse.ArgumentParser(description="Evaluate a saved U-Net checkpoint for HW2 Task 3.")  # 创建命令行解析器。
    # 下面这个代码块添加 checkpoint 路径参数。
    parser.add_argument(  # 添加一个命令行参数。
        "--checkpoint",  # 参数名，使用示例是 --checkpoint checkpoints/unet_ce_best.pth。
        required=True,  # checkpoint 是必须传入的，因为评估必须知道加载哪个模型权重。
        help="要评估的 .pth checkpoint 路径，建议使用 checkpoints/unet_ce_best.pth 这类项目内路径。",  # 参数帮助信息。
    )  # checkpoint 参数添加结束。
    # 下面这个代码块添加数据集划分参数。
    parser.add_argument(  # 添加一个命令行参数。
        "--split",  # 参数名，使用示例是 --split val。
        choices=["val", "test"],  # 本评估脚本只支持验证集和测试集，避免误把训练集当最终评估结果。
        default="val",  # 默认评估验证集，因为要求比较验证集 mIoU。
        help="选择评估 val 还是 test split。",  # 参数帮助信息。
    )  # split 参数添加结束。
    # 下面这个代码块添加 batch size 参数。
    parser.add_argument(  # 添加一个命令行参数。
        "--batch-size",  # 参数名，使用示例是 --batch-size 2。
        type=int,  # batch size 必须是整数。
        default=2,  # 默认使用 2，和当前短测训练设置保持一致。
        help="评估时每个 batch 的图像数量。",  # 参数帮助信息。
    )  # batch-size 参数添加结束。
    # 下面这个代码块添加最多评估多少个 batch 的参数。
    parser.add_argument(  # 添加一个命令行参数。
        "--max-batches",  # 参数名，使用示例是 --max-batches 2。
        type=int,  # batch 数量必须是整数。
        default=2,  # 默认只评估 2 个 batch，适合作为最小短测。
        help="最多评估多少个 batch；小于等于 0 表示跑完整个 split。",  # 参数帮助信息。
    )  # max-batches 参数添加结束。
    # 下面这个代码块添加随机种子参数。
    parser.add_argument(  # 添加一个命令行参数。
        "--seed",  # 参数名，使用示例是 --seed 42。
        type=int,  # 随机种子必须是整数。
        default=42,  # 默认使用 42，和 dataset.py 与 train.py 的划分保持一致。
        help="数据集划分使用的随机种子，应该与训练时保持一致。",  # 参数帮助信息。
    )  # seed 参数添加结束。
    # 下面这个代码块返回解析后的参数对象。
    return parser.parse_args()  # 返回 argparse.Namespace，后续 main 函数会读取它。


# 下面这个函数负责把传入的 checkpoint 路径解析成项目内部的绝对路径。
def resolve_project_input_path(project_root, path_text):  # 定义项目内输入路径解析函数。
    # 下面这个代码块把字符串路径转换成 Path 对象。
    raw_path = Path(path_text)  # raw_path 可能是相对路径，也可能是绝对路径。
    # 下面这个代码块把相对路径解释为相对于项目根目录的路径。
    if not raw_path.is_absolute():  # 如果传入的是相对路径。
        raw_path = project_root / raw_path  # 就把它拼接到项目根目录下面。
    # 下面这个代码块解析项目根目录的真实绝对路径。
    resolved_project_root = project_root.resolve()  # resolve 可以消除路径中的 .. 等部分。
    # 下面这个代码块解析 checkpoint 的真实绝对路径。
    resolved_path = raw_path.resolve()  # 得到最终准备读取的 checkpoint 路径。
    # 下面这个代码块检查 checkpoint 路径是否仍然位于当前项目目录内部。
    if resolved_path != resolved_project_root and resolved_project_root not in resolved_path.parents:  # 如果路径跑到项目外部。
        raise ValueError(f"checkpoint 路径必须位于当前项目目录内部: {resolved_path}")  # 直接报错，避免读取项目外文件。
    # 下面这个代码块检查 checkpoint 文件是否存在。
    if not resolved_path.is_file():  # 如果目标路径不是一个真实文件。
        raise FileNotFoundError(f"找不到 checkpoint 文件: {resolved_path}")  # 给出清晰的缺失文件提示。
    # 下面这个代码块返回安全可用的 checkpoint 路径。
    return resolved_path  # 返回 Path 对象，供 torch.load 使用。


# 下面这个函数负责从 checkpoint 字典里读取必须存在的键。
def get_required_checkpoint_value(checkpoint, key):  # 定义 checkpoint 必需字段读取函数。
    # 下面这个代码块检查 checkpoint 是否包含指定字段。
    if key not in checkpoint:  # 如果 checkpoint 里没有这个键。
        raise KeyError(f"checkpoint 缺少必要字段: {key}")  # 直接报错，说明权重文件格式不符合当前 train.py。
    # 下面这个代码块返回 checkpoint 中对应的值。
    return checkpoint[key]  # 返回字段值，供模型重建或评估使用。


# 下面这个函数负责加载 .pth checkpoint 文件。
def load_checkpoint(checkpoint_path, device):  # 定义 checkpoint 加载函数。
    # 下面这个代码块尝试使用较新的 torch.load 参数加载 checkpoint。
    try:  # 新版 PyTorch 支持 weights_only 参数。
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # 加载完整 checkpoint 字典。
    # 下面这个代码块兼容不支持 weights_only 参数的旧版 PyTorch。
    except TypeError:  # 如果当前 PyTorch 版本不认识 weights_only 参数。
        checkpoint = torch.load(checkpoint_path, map_location=device)  # 就使用传统写法加载 checkpoint。
    # 下面这个代码块检查加载结果是否为字典。
    if not isinstance(checkpoint, dict):  # train.py 保存的 checkpoint 应该是一个 dict。
        raise TypeError("checkpoint 内容应为字典，请确认它来自当前项目的 src/train.py。")  # 给出清楚的格式错误提示。
    # 下面这个代码块返回加载后的 checkpoint。
    return checkpoint  # 返回 checkpoint 字典。


# 下面这个函数负责根据 checkpoint 中保存的结构信息重建 U-Net。
def build_model_from_checkpoint(checkpoint, device):  # 定义模型重建函数。
    # 下面这个代码块读取类别数。
    num_classes = int(get_required_checkpoint_value(checkpoint, "num_classes"))  # num_classes 应该是 8。
    # 下面这个代码块读取 U-Net 的通道配置。
    features = tuple(get_required_checkpoint_value(checkpoint, "features"))  # features 例如是 (16, 32, 64, 128)。
    # 下面这个代码块读取模型参数字典。
    model_state_dict = get_required_checkpoint_value(checkpoint, "model_state_dict")  # model_state_dict 保存模型权重。
    # 下面这个代码块创建和训练时结构一致的 U-Net。
    model = UNet(in_channels=3, num_classes=num_classes, features=features)  # 输入通道为 RGB 的 3，输出通道为类别数。
    # 下面这个代码块加载 checkpoint 中的模型权重。
    model.load_state_dict(model_state_dict)  # 把保存好的权重装入模型。
    # 下面这个代码块把模型移动到 CPU 或 GPU。
    model = model.to(device)  # 让模型和后续输入 batch 位于同一个设备。
    # 下面这个代码块把模型切换为评估模式。
    model.eval()  # 评估模式会固定 BatchNorm 等层的行为。
    # 下面这个代码块返回模型和结构信息。
    return model, num_classes, features  # 返回重建后的模型、类别数和通道配置。


# 下面这个函数负责根据 checkpoint 中保存的 loss_name 重建 loss。
def build_loss_from_checkpoint(checkpoint, num_classes, device):  # 定义 loss 重建函数。
    # 下面这个代码块读取训练时使用的 loss 名称。
    loss_name = str(get_required_checkpoint_value(checkpoint, "loss_name"))  # loss_name 应该是 ce、dice 或 ce_dice。
    # 下面这个代码块读取训练时使用的 ignore_index。
    ignore_index = int(get_required_checkpoint_value(checkpoint, "ignore_index"))  # ignore_index 当前应为 -1。
    # 下面这个代码块根据 loss 名称创建对应的 loss 模块。
    loss_fn = get_loss_function(loss_name, num_classes=num_classes, ignore_index=ignore_index)  # 重建训练时同类型的 loss。
    # 下面这个代码块把 loss 模块移动到目标设备。
    loss_fn = loss_fn.to(device)  # 让 loss 计算和 logits、mask 位于同一设备。
    # 下面这个代码块返回 loss 模块和相关设置。
    return loss_fn, loss_name, ignore_index  # 返回 loss 函数、loss 名称和 ignore_index。


# 下面这个函数负责创建 val 或 test 的 DataLoader。
def build_evaluation_loader(dataset_root, split, batch_size, seed):  # 定义评估 DataLoader 构造函数。
    # 下面这个代码块构造指定 split 的 Dataset。
    dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split=split, seed=seed)  # 读取 val 或 test 数据。
    # 下面这个代码块把 Dataset 包装成 DataLoader。
    loader = DataLoader(  # 创建 PyTorch DataLoader。
        dataset,  # 传入要评估的数据集。
        batch_size=batch_size,  # 每个 batch 读取多少张图像。
        shuffle=False,  # 评估阶段不打乱顺序，方便复现和排查问题。
        num_workers=0,  # 使用 0 更稳，不额外开子进程。
        collate_fn=segmentation_collate_fn,  # 复用训练时的 padding 规则处理不同尺寸图像。
    )  # DataLoader 创建结束。
    # 下面这个代码块返回 DataLoader。
    return loader  # 返回可迭代的评估数据加载器。


# 下面这个函数负责在指定 split 上运行评估循环。
def evaluate_model(model, loader, loss_fn, device, num_classes, ignore_index, max_batches, split):  # 定义模型评估函数。
    # 下面这个代码块把模型切换到评估模式。
    model.eval()  # 确保评估时 BatchNorm 等层使用评估行为。
    # 下面这个代码块初始化 loss 累计值。
    total_loss = 0.0  # 用来累加每个 batch 的 loss。
    # 下面这个代码块初始化已经处理的 batch 数量。
    total_batches = 0  # 用来计算平均 loss。
    # 下面这个代码块创建指标累计器。
    tracker = SegmentationMetricsTracker(num_classes=num_classes, ignore_index=ignore_index)  # 用来累计 accuracy 和 mIoU。
    # 下面这个代码块关闭梯度记录，因为评估不需要反向传播。
    with torch.no_grad():  # 关闭梯度可以节省显存和时间。
        # 下面这个代码块逐个读取评估 batch。
        for batch_index, (images, masks) in enumerate(loader, start=1):  # batch_index 从 1 开始，打印时更直观。
            # 下面这个代码块把图像和 mask 移动到目标设备。
            images, masks = move_batch_to_device(images, masks, device)  # images 是 [B, 3, H, W]，masks 是 [B, H, W]。
            # 下面这个代码块执行 U-Net 前向传播。
            logits = model(images)  # logits 的形状应该是 [B, num_classes, H, W]。
            # 下面这个代码块计算当前 batch 的评估 loss。
            loss = loss_fn(logits, masks)  # loss 是标量 tensor，不会反向传播。
            # 下面这个代码块累计 loss 数值。
            total_loss += loss.item()  # 把 tensor 标量转成 Python 数字并累加。
            # 下面这个代码块累计 batch 数量。
            total_batches += 1  # 记录已经评估了一个 batch。
            # 下面这个代码块更新像素准确率和 mIoU 统计。
            tracker.update(logits, masks)  # 用 logits 和真实 mask 统计当前 batch 的分割指标。
            # 下面这个代码块打印当前 batch 的简短日志。
            print(f"{split} batch {batch_index}: loss={loss.item():.6f}")  # 打印当前 batch 的 loss。
            # 下面这个代码块判断是否达到短测 batch 上限。
            if reached_batch_limit(batch_index, max_batches):  # max_batches 小于等于 0 时不会提前停止。
                break  # 达到短测上限后停止评估循环。
    # 下面这个代码块检查是否真的评估过至少一个 batch。
    if total_batches == 0:  # 如果没有处理任何 batch，说明数据集或参数有问题。
        raise RuntimeError("评估阶段没有处理任何 batch。")  # 给出清晰的错误提示。
    # 下面这个代码块计算平均评估 loss。
    average_loss = total_loss / total_batches  # 用累计 loss 除以 batch 数量。
    # 下面这个代码块计算累计指标。
    metrics = tracker.compute()  # 得到 pixel_accuracy 和 miou。
    # 下面这个代码块把 loss 和 batch 数量加入结果字典。
    metrics["eval_loss"] = average_loss  # 保存平均评估 loss。
    # 下面这个代码块把实际评估的 batch 数量加入结果字典。
    metrics["evaluated_batches"] = total_batches  # 保存短测或完整评估实际跑了多少个 batch。
    # 下面这个代码块返回最终评估结果。
    return metrics  # 返回包含 eval_loss、pixel_accuracy、miou、evaluated_batches 的字典。


# 下面这个函数是评估脚本的主入口，负责串联参数、路径、checkpoint、模型、数据和评估循环。
def main():  # 定义主函数。
    # 下面这个代码块读取命令行参数。
    args = parse_args()  # 获取用户在 PowerShell 中传入的参数。
    # 下面这个代码块检查 batch size 是否有效。
    if args.batch_size <= 0:  # batch size 必须是正整数。
        raise ValueError("batch-size 必须大于 0。")  # 给出清晰的参数错误提示。
    # 下面这个代码块定位当前项目根目录。
    project_root = Path(__file__).resolve().parents[1]  # evaluate.py 位于 src 下，所以 parents[1] 是项目根目录。
    # 下面这个代码块拼接 Stanford Background Dataset 的项目内路径。
    dataset_root = project_root / "data" / "stanford_background" / "iccv09Data"  # 数据集主体目录。
    # 下面这个代码块解析并检查 checkpoint 路径。
    checkpoint_path = resolve_project_input_path(project_root, args.checkpoint)  # 确保 checkpoint 位于项目目录内部。
    # 下面这个代码块选择评估设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 就使用 GPU，否则使用 CPU。
    # 下面这个代码块加载 checkpoint。
    checkpoint = load_checkpoint(checkpoint_path, device)  # 读取 train.py 保存的 .pth 文件。
    # 下面这个代码块根据 checkpoint 重建 U-Net。
    model, num_classes, features = build_model_from_checkpoint(checkpoint, device)  # 得到已加载权重的模型。
    # 下面这个代码块根据 checkpoint 重建 loss。
    loss_fn, loss_name, ignore_index = build_loss_from_checkpoint(checkpoint, num_classes, device)  # 得到评估使用的 loss。
    # 下面这个代码块构造评估数据加载器。
    loader = build_evaluation_loader(dataset_root, args.split, args.batch_size, args.seed)  # 创建 val 或 test DataLoader。
    # 下面这个代码块读取 checkpoint 中的 epoch，用于打印说明。
    checkpoint_epoch = checkpoint.get("epoch", "unknown")  # 如果旧 checkpoint 没有 epoch，就显示 unknown。
    # 下面这个代码块读取 checkpoint 中记录的最好验证 mIoU，用于打印参考信息。
    checkpoint_best_miou = checkpoint.get("best_val_miou", "unknown")  # 如果旧 checkpoint 没有该字段，就显示 unknown。
    # 下面这个代码块打印本次评估的基础配置。
    print(f"device: {device}")  # 打印使用 CPU 还是 GPU。
    # 下面这个代码块打印 checkpoint 路径。
    print(f"checkpoint_path: {checkpoint_path}")  # 打印当前加载的模型权重文件。
    # 下面这个代码块打印 checkpoint 对应的训练 epoch。
    print(f"checkpoint_epoch: {checkpoint_epoch}")  # 打印 checkpoint 来自第几个 epoch。
    # 下面这个代码块打印 checkpoint 中记录的最佳验证 mIoU。
    print(f"checkpoint_best_val_miou: {checkpoint_best_miou}")  # 打印训练阶段记录的最佳验证 mIoU。
    # 下面这个代码块打印数据集路径。
    print(f"dataset_root: {dataset_root}")  # 打印数据集读取位置。
    # 下面这个代码块打印评估 split 和样本数量。
    print(f"split: {args.split}, samples: {len(loader.dataset)}")  # 打印本次评估的数据划分和样本数。
    # 下面这个代码块打印评估 batch 设置。
    print(f"batch_size: {args.batch_size}, max_batches: {args.max_batches}")  # 打印 batch size 和短测 batch 限制。
    # 下面这个代码块打印模型结构关键信息。
    print(f"num_classes: {num_classes}, ignore_index: {ignore_index}, features: {features}")  # 打印类别数、忽略标签和通道配置。
    # 下面这个代码块打印 loss 配置。
    print(f"loss: {loss_name}")  # 打印 checkpoint 记录的 loss 配置。
    # 下面这个代码块真正执行评估。
    metrics = evaluate_model(model, loader, loss_fn, device, num_classes, ignore_index, args.max_batches, args.split)  # 运行评估循环。
    # 下面这个代码块打印最终评估结果。
    print(f"{args.split} summary: eval_loss={metrics['eval_loss']:.6f}, pixel_accuracy={metrics['pixel_accuracy']:.6f}, miou={metrics['miou']:.6f}, evaluated_batches={metrics['evaluated_batches']}")  # 打印汇总指标。
    # 下面这个代码块打印评估完成提示。
    print("evaluation finished.")  # 表示评估脚本已经正常结束。


# 下面这个入口保证只有直接运行 python src/evaluate.py 时才会启动评估。
if __name__ == "__main__":  # 判断当前文件是否作为脚本直接运行。
    main()  # 调用主函数，开始评估流程。
