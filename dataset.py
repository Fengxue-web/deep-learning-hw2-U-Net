# 这个文件实现 Stanford Background Dataset 的最小 PyTorch 读取模块。

# 导入标准库 Path，用来安全地拼接和检查项目内的数据路径。
from pathlib import Path  # Path 可以让 Windows 路径和普通路径写法都更清晰。

# 导入标准库 random，用来做固定随机种子的 train/val/test 划分。
import random  # random.Random(seed) 可以保证每次划分结果一致。

# 导入类型提示。
from typing import List, Sequence, Tuple  # 这些类型只帮助阅读，不改变运行逻辑。

# 导入 PyTorch，本项目后续训练、loss 和 DataLoader 都会使用 torch.Tensor。
import torch  # torch 用来把图像和 mask 转成模型可以处理的 tensor。

# 导入 PyTorch 的 Dataset 基类，方便后续交给 DataLoader 使用。
from torch.utils.data import Dataset  # Dataset 是所有自定义 PyTorch 数据集的常用父类。

# 尝试导入 Pillow 的 Image，用来读取 jpg 图像文件。
PIL_IMPORT_ERROR = None  # 如果 Pillow 没有安装，这里会保存导入失败的错误信息。

# 这个代码块负责导入图像读取工具，如果失败，会在真正读取图片时给出清楚提示。
try:  # 尝试导入 Pillow，因为 Python 标准库不能直接解码 jpg 图像。
    from PIL import Image  # Image.open(...) 可以读取 Stanford 数据集里的 jpg 图像。
except ImportError as error:  # 如果当前环境没有 Pillow，就捕获这个导入错误。
    Image = None  # 用 None 表示图像读取工具当前不可用。
    PIL_IMPORT_ERROR = error  # 保存原始错误，方便后面报错时说明原因。

# 这个代码块定义数据集的基本常量，后续 loss 和 metrics 会用到这些约定。
NUM_CLASSES = 8  # regions.txt 中有效语义类别编号为 0 到 7，所以类别数是 8。
IGNORE_INDEX = -1  # 负数表示 unknown，训练 loss 中应忽略这个标签。
CLASS_NAMES = (  # 这些类别名称来自数据集 README 对 regions.txt 的说明。
    "sky",  # 类别 0：天空。
    "tree",  # 类别 1：树木。
    "road",  # 类别 2：道路。
    "grass",  # 类别 3：草地。
    "water",  # 类别 4：水面。
    "building",  # 类别 5：建筑。
    "mountain",  # 类别 6：山体。
    "foreground_object",  # 类别 7：前景物体。
)  # CLASS_NAMES 的顺序要和标签编号 0 到 7 对齐。


# 这个函数负责扫描 images 和 labels 目录，建立图像文件与语义 mask 文件的一一对应关系。
def build_image_mask_pairs(root_dir: Path) -> List[Tuple[Path, Path]]:  # 返回值是若干个 (image_path, mask_path) 元组。
    image_dir = root_dir / "images"  # Stanford 图像文件所在目录。
    label_dir = root_dir / "labels"  # Stanford 标注文件所在目录。

    # 这个代码块检查 images 目录是否存在，避免后面报更难懂的错误。
    if not image_dir.is_dir():  # 如果 images 目录不存在，说明数据路径可能不对。
        raise FileNotFoundError(f"找不到图像目录: {image_dir}")  # 直接说明缺少哪个目录。

    # 这个代码块检查 labels 目录是否存在，避免 mask 路径拼错后静默失败。
    if not label_dir.is_dir():  # 如果 labels 目录不存在，说明数据集可能没有正确解压。
        raise FileNotFoundError(f"找不到标签目录: {label_dir}")  # 直接说明缺少哪个目录。

    # 这个代码块读取所有 jpg 图像，并按文件名排序，保证每次运行顺序一致。
    image_paths = sorted(image_dir.glob("*.jpg"))  # Stanford Background Dataset 的图像格式是 jpg。
    pairs = []  # 这里保存最终找到的图像和 mask 路径配对。

    # 这个代码块为每张图像查找同名的 .regions.txt 语义分割标签。
    for image_path in image_paths:  # 逐张处理已经排序好的图像路径。
        mask_name = f"{image_path.stem}.regions.txt"  # 例如 0000047.jpg 对应 0000047.regions.txt。
        mask_path = label_dir / mask_name  # 拼出当前图像对应的语义标签路径。
        if not mask_path.is_file():  # 如果找不到对应 mask，说明数据不完整或文件名不匹配。
            raise FileNotFoundError(f"图像缺少对应的 regions 标签: {image_path}")  # 明确指出是哪张图像缺标签。
        pairs.append((image_path, mask_path))  # 保存这一对图像和 mask 路径。

    # 这个代码块检查是否真的找到了样本，避免空数据集继续往下运行。
    if len(pairs) == 0:  # 如果没有任何 jpg 图像，就无法构造数据集。
        raise RuntimeError(f"没有在目录中找到 jpg 图像: {image_dir}")  # 给出清楚的空数据集报错。

    return pairs  # 返回所有图像和 mask 的配对结果。


# 这个函数负责在没有官方 split 文件时，做一个固定随机种子的 train/val/test 划分。
def split_pairs(pairs: Sequence[Tuple[Path, Path]], split: str, seed: int = 42) -> List[Tuple[Path, Path]]:  # split 可以是 train、val、test 或 all。
    allowed_splits = {"train", "val", "test", "all"}  # 这里列出本文件支持的划分名称。

    # 这个代码块检查 split 参数，避免写错名称后得到奇怪结果。
    if split not in allowed_splits:  # 如果 split 不在允许列表里，就说明参数写错了。
        raise ValueError(f"split 必须是 {allowed_splits} 之一，但收到: {split}")  # 报错时直接展示合法选项。

    # 这个代码块支持 all 模式，方便一次性检查完整数据集数量。
    if split == "all":  # all 表示不做划分，直接使用全部样本。
        return list(pairs)  # 返回一个新的列表，避免外部意外修改原始 pairs。

    shuffled_pairs = list(pairs)  # 复制一份样本列表，避免改变外部传入的原始顺序。
    rng = random.Random(seed)  # 创建固定随机种子的随机数生成器。
    rng.shuffle(shuffled_pairs)  # 用固定种子打乱样本，保证 train/val/test 每次一致。

    total_count = len(shuffled_pairs)  # 统计总样本数，当前数据集应为 715。
    train_count = int(total_count * 0.8)  # 训练集使用 80% 样本。
    val_count = int(total_count * 0.1)  # 验证集使用 10% 样本。
    train_end = train_count  # train 的结束位置，也是 val 的开始位置。
    val_end = train_count + val_count  # val 的结束位置，也是 test 的开始位置。

    # 这个代码块根据 split 名称返回对应的一段样本列表。
    if split == "train":  # 如果请求训练集。
        return shuffled_pairs[:train_end]  # 返回打乱后前 80% 的样本。
    if split == "val":  # 如果请求验证集。
        return shuffled_pairs[train_end:val_end]  # 返回打乱后中间 10% 的样本。
    return shuffled_pairs[val_end:]  # 剩余样本作为测试集。


# 这个函数负责把 jpg 图像读取为 PyTorch tensor。
def load_image_as_tensor(image_path: Path) -> torch.Tensor:  # 返回形状为 [3, H, W] 的 float32 tensor。
    # 这个代码块检查 Pillow 是否可用，因为读取 jpg 需要图像解码库。
    if Image is None:  # 如果 Pillow 没有成功导入，就不能读取 jpg。
        raise RuntimeError(f"读取 jpg 需要 Pillow，但当前导入失败: {PIL_IMPORT_ERROR}")  # 给出明确依赖提示。

    # 这个代码块打开图像并统一转为 RGB，保证通道数固定为 3。
    with Image.open(image_path) as image:  # 使用 with 可以在读取后自动关闭文件。
        rgb_image = image.convert("RGB")  # 把图像统一转换为 RGB 三通道。
        width, height = rgb_image.size  # PIL 的 size 顺序是 (宽度, 高度)。
        image_bytes = bytearray(rgb_image.tobytes())  # 把 RGB 图像转成连续的字节数据。

    # 这个代码块把图像字节转换为模型常用的 [C, H, W] tensor。
    image_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)  # 先得到一维 uint8 tensor。
    image_tensor = image_tensor.view(height, width, 3)  # 还原为 [H, W, 3] 的图像形状。
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # 转成 PyTorch 常用的 [3, H, W]。
    image_tensor = image_tensor.float().div(255.0)  # 转为 float32，并把像素范围缩放到 [0, 1]。

    return image_tensor  # 返回可以直接输入模型的图像 tensor。


# 这个函数负责把 .regions.txt 语义标签读取为 PyTorch tensor。
def load_mask_as_tensor(mask_path: Path) -> torch.Tensor:  # 返回形状为 [H, W] 的 long tensor。
    rows = []  # 这里保存从文本文件中逐行读出的整数标签。

    # 这个代码块逐行读取 regions.txt，每一行对应 mask 的一行像素标签。
    with mask_path.open("r", encoding="utf-8") as file:  # 用文本模式读取标签矩阵。
        for line in file:  # 逐行读取文本文件。
            clean_line = line.strip()  # 去掉行首行尾空白字符。
            if clean_line == "":  # 如果遇到空行，就跳过它。
                continue  # 空行不代表有效像素标签。
            row = [int(value) for value in clean_line.split()]  # 把一行空格分隔的数字转成整数列表。
            rows.append(row)  # 把这一行标签加入完整 mask。

    # 这个代码块检查 mask 文件是否为空。
    if len(rows) == 0:  # 空 mask 无法用于训练或验证。
        raise RuntimeError(f"mask 文件为空: {mask_path}")  # 明确指出哪个文件为空。

    expected_width = len(rows[0])  # 用第一行长度作为期望宽度。

    # 这个代码块检查每一行宽度是否一致，保证可以变成规则二维矩阵。
    for row_index, row in enumerate(rows):  # 逐行检查读取到的 mask。
        if len(row) != expected_width:  # 如果某一行长度不同，说明标签文件格式异常。
            raise ValueError(f"mask 第 {row_index} 行宽度不一致: {mask_path}")  # 报出异常文件和行号。

    mask_tensor = torch.tensor(rows, dtype=torch.long)  # CrossEntropyLoss 需要 long 类型标签。

    return mask_tensor  # 返回形状为 [H, W] 的 mask tensor。


# 这个类把 Stanford Background Dataset 封装成 PyTorch Dataset。
class StanfordBackgroundDataset(Dataset):  # 继承 Dataset 后，就可以被 DataLoader 使用。
    num_classes = NUM_CLASSES  # 数据集有效语义类别数为 8。
    ignore_index = IGNORE_INDEX  # unknown 标签为 -1，后续 loss 应忽略它。
    class_names = CLASS_NAMES  # 保存类别名称，方便调试和写报告。

    # 这个代码块初始化数据集路径、划分方式和样本列表。
    def __init__(self, root_dir: str = "data/stanford_background/iccv09Data", split: str = "train", seed: int = 42) -> None:  # root_dir 默认指向本项目数据目录。
        self.root_dir = Path(root_dir)  # 把字符串路径转换成 Path 对象。
        self.split = split  # 保存当前使用的是 train、val、test 还是 all。
        self.seed = seed  # 保存随机种子，保证划分可以复现。
        self.all_pairs = build_image_mask_pairs(self.root_dir)  # 扫描全部图像和 regions mask 配对。
        self.samples = split_pairs(self.all_pairs, split=self.split, seed=self.seed)  # 按 split 取出当前子集。

    # 这个代码块返回当前 split 中有多少个样本。
    def __len__(self) -> int:  # PyTorch Dataset 必须实现 __len__。
        return len(self.samples)  # 返回当前子集的样本数量。

    # 这个代码块根据 index 读取一张图像和它对应的语义 mask。
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # PyTorch Dataset 必须实现 __getitem__。
        image_path, mask_path = self.samples[index]  # 根据索引取出图像路径和 mask 路径。
        image_tensor = load_image_as_tensor(image_path)  # 读取图像，得到 [3, H, W] float32 tensor。
        mask_tensor = load_mask_as_tensor(mask_path)  # 读取语义标签，得到 [H, W] long tensor。

        # 这个代码块检查图像空间尺寸和 mask 空间尺寸是否一致。
        if image_tensor.shape[1:] != mask_tensor.shape:  # 图像的 H,W 必须等于 mask 的 H,W。
            raise ValueError(f"图像和 mask 尺寸不一致: {image_path}, {mask_path}")  # 如果不一致，直接报错避免错误训练。

        return image_tensor, mask_tensor  # 返回模型输入图像和监督标签。

    # 这个代码块提供样本路径查询，方便调试时知道当前 index 对应哪个文件。
    def get_sample_paths(self, index: int) -> Tuple[Path, Path]:  # 返回指定样本的 image_path 和 mask_path。
        return self.samples[index]  # 直接返回保存好的路径配对。


# 这个函数实现最小自测，运行 python src/dataset.py 时会执行它。
def main() -> None:  # main 不参与训练，只用于检查数据读取是否正确。
    project_root = Path(__file__).resolve().parents[1]  # dataset.py 位于 src 下，所以 parents[1] 是项目根目录。
    dataset_root = project_root / "data" / "stanford_background" / "iccv09Data"  # 拼出 Stanford 数据集主体目录。

    # 这个代码块分别构造 train、val、test 和 all 数据集，检查划分数量。
    train_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="train")  # 构造训练集。
    val_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="val")  # 构造验证集。
    test_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="test")  # 构造测试集。
    all_dataset = StanfordBackgroundDataset(root_dir=str(dataset_root), split="all")  # 构造完整数据集。

    # 这个代码块打印数据集路径和样本数量，确认数据是否被正确扫描。
    print(f"dataset_root: {dataset_root}")  # 打印当前读取的数据集目录。
    print(f"all samples: {len(all_dataset)}")  # 打印全部样本数量，预期为 715。
    print(f"train samples: {len(train_dataset)}")  # 打印训练集样本数量，预期约为 572。
    print(f"val samples: {len(val_dataset)}")  # 打印验证集样本数量，预期约为 71。
    print(f"test samples: {len(test_dataset)}")  # 打印测试集样本数量，预期约为 72。

    # 这个代码块读取训练集第一个样本，检查 image 和 mask 的 tensor 形状。
    sample_index = 0  # 只读取第一个样本，避免做耗时的大规模检查。
    image_path, mask_path = train_dataset.get_sample_paths(sample_index)  # 获取第一个样本对应的文件路径。
    image_tensor, mask_tensor = train_dataset[sample_index]  # 真正读取第一个样本的图像和 mask。
    unique_labels = torch.unique(mask_tensor).tolist()  # 统计这个 mask 中出现过的标签编号。

    # 这个代码块打印第一个样本的详细信息，帮助确认 tensor 是否符合后续训练要求。
    print(f"sample image path: {image_path}")  # 打印样本图像路径。
    print(f"sample mask path: {mask_path}")  # 打印样本 mask 路径。
    print(f"image shape: {tuple(image_tensor.shape)}")  # 预期格式为 (3, H, W)。
    print(f"image dtype: {image_tensor.dtype}")  # 预期为 torch.float32。
    print(f"image min/max: {image_tensor.min().item():.4f}/{image_tensor.max().item():.4f}")  # 预期范围在 0 到 1。
    print(f"mask shape: {tuple(mask_tensor.shape)}")  # 预期格式为 (H, W)。
    print(f"mask dtype: {mask_tensor.dtype}")  # 预期为 torch.int64，也就是 long。
    print(f"mask unique labels in sample: {unique_labels}")  # 标签可能包含 -1，也可能包含 0 到 7。
    print(f"num_classes: {NUM_CLASSES}")  # 打印有效类别数量。
    print(f"ignore_index: {IGNORE_INDEX}")  # 打印后续 loss 应忽略的 unknown 标签。


# 这个代码块保证只有直接运行本文件时才执行最小自测。
if __name__ == "__main__":  # 如果命令是 python src/dataset.py，就会进入这里。
    main()  # 执行最小数据读取测试。
