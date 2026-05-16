# 本文件实现语义分割任务需要用到的损失函数。
# 这里使用 PyTorch，并手动实现 Dice Loss。

# 导入 PyTorch 主库，用来创建 tensor、做 softmax、检查梯度等操作。
import torch  # 导入 PyTorch 主模块。

# 导入 torch.nn，用来继承 nn.Module，并使用标准 CrossEntropyLoss。
import torch.nn as nn  # 导入神经网络模块并命名为 nn。

# 导入 torch.nn.functional，用来调用 one_hot 等函数式工具。
import torch.nn.functional as F  # 导入函数式 API 并命名为 F。


# 下面这个类把 PyTorch 自带的 CrossEntropyLoss 包装成语义分割 loss。
class CrossEntropySegmentationLoss(nn.Module):  # 定义语义分割交叉熵损失类。
    # 初始化交叉熵损失，并记录需要忽略的标签值。
    def __init__(self, ignore_index=-1):  # 定义初始化函数，默认忽略标签 -1。
        super().__init__()  # 调用父类 nn.Module 的初始化函数。
        self.ignore_index = ignore_index  # 保存 ignore_index，方便之后查看当前设置。
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)  # 创建 PyTorch 标准交叉熵损失。

    # 前向计算交叉熵损失。
    def forward(self, logits, targets):  # logits 形状是 [B, C, H, W]，targets 形状是 [B, H, W]。
        loss = self.loss_fn(logits, targets)  # 计算交叉熵，并自动忽略 targets 中等于 ignore_index 的像素。
        return loss  # 返回标量 loss。


# 下面这个类手动实现多类别 Dice Loss。
class DiceLoss(nn.Module):  # 定义 Dice Loss 类。
    # 初始化 Dice Loss 需要的类别数、忽略标签和平滑项。
    def __init__(self, num_classes, ignore_index=-1, smooth=1e-6):  # 定义初始化函数。
        super().__init__()  # 调用父类 nn.Module 的初始化函数。
        self.num_classes = num_classes  # 保存类别数，例如 Stanford Background Dataset 当前记录为 8 类。
        self.ignore_index = ignore_index  # 保存需要忽略的标签值，例如 unknown 标签为 -1。
        self.smooth = smooth  # 保存平滑项，避免分母为 0。

    # 前向计算 Dice Loss。
    def forward(self, logits, targets):  # logits 形状是 [B, C, H, W]，targets 形状是 [B, H, W]。
        if logits.dim() != 4:  # 检查 logits 是否有 batch、class、height、width 四个维度。
            raise ValueError("logits 应该是 [B, C, H, W] 形状。")  # 如果形状不对，给出清晰报错。
        if targets.dim() != 3:  # 检查 targets 是否有 batch、height、width 三个维度。
            raise ValueError("targets 应该是 [B, H, W] 形状。")  # 如果形状不对，给出清晰报错。
        if logits.shape[1] != self.num_classes:  # 检查 logits 的类别通道数是否等于 num_classes。
            raise ValueError("logits 的类别通道数必须等于 num_classes。")  # 如果类别数不一致，给出清晰报错。

        valid_mask = targets != self.ignore_index  # 得到有效像素位置，形状是 [B, H, W]。
        if valid_mask.sum() == 0:  # 如果一个 batch 里所有像素都需要忽略。
            return logits.sum() * 0.0  # 返回一个可反向传播的 0，避免产生 nan。

        safe_targets = targets.clone()  # 复制 targets，避免直接修改原始标签 tensor。
        safe_targets[~valid_mask] = 0  # 把忽略像素临时改成 0，保证 one_hot 不会因为 -1 报错。

        probabilities = torch.softmax(logits, dim=1)  # 对类别维度做 softmax，得到每个像素属于每一类的概率。
        one_hot_targets = F.one_hot(safe_targets, num_classes=self.num_classes)  # 把 mask 转成 one-hot，形状是 [B, H, W, C]。
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).float()  # 调整成 [B, C, H, W]，并转成浮点数。

        valid_mask = valid_mask.unsqueeze(1).float()  # 把有效像素 mask 调整成 [B, 1, H, W]，方便和类别维度广播。
        probabilities = probabilities * valid_mask  # 忽略像素位置的预测概率置为 0。
        one_hot_targets = one_hot_targets * valid_mask  # 忽略像素位置的真实标签 one-hot 置为 0。

        reduce_dims = (0, 2, 3)  # Dice 对 batch、高度、宽度求和，只保留类别维度。
        intersection = torch.sum(probabilities * one_hot_targets, dim=reduce_dims)  # 计算每一类预测和标签的交集。
        probability_sum = torch.sum(probabilities, dim=reduce_dims)  # 计算每一类预测概率总和。
        target_sum = torch.sum(one_hot_targets, dim=reduce_dims)  # 计算每一类真实标签像素总和。

        dice_per_class = (2.0 * intersection + self.smooth) / (probability_sum + target_sum + self.smooth)  # 计算每一类 Dice。
        class_has_pixels = target_sum > 0  # 只统计当前 batch 中真实出现过的类别，避免空类别干扰平均值。
        dice_mean = dice_per_class[class_has_pixels].mean()  # 对出现过的类别求平均 Dice。
        loss = 1.0 - dice_mean  # Dice Loss 等于 1 减去平均 Dice。
        return loss  # 返回标量 Dice Loss。


# 下面这个类实现组合损失：Cross-Entropy Loss + Dice Loss。
class CombinedLoss(nn.Module):  # 定义组合损失类。
    # 初始化组合损失内部需要的 CE 和 Dice。
    def __init__(self, num_classes, ignore_index=-1):  # 定义初始化函数。
        super().__init__()  # 调用父类 nn.Module 的初始化函数。
        self.ce_loss = CrossEntropySegmentationLoss(ignore_index=ignore_index)  # 创建交叉熵损失模块。
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)  # 创建 Dice Loss 模块。

    # 前向计算组合损失。
    def forward(self, logits, targets):  # logits 形状是 [B, C, H, W]，targets 形状是 [B, H, W]。
        ce = self.ce_loss(logits, targets)  # 计算交叉熵损失。
        dice = self.dice_loss(logits, targets)  # 计算 Dice Loss。
        loss = ce + dice  # 把两种损失直接相加，形成组合损失。
        return loss  # 返回标量组合损失。


# 下面这个函数根据字符串名称创建对应的 loss，方便 train.py 后续调用。
def get_loss_function(loss_name, num_classes, ignore_index=-1):  # 定义 loss 工厂函数。
    normalized_name = loss_name.lower()  # 把 loss 名称转成小写，减少大小写输入带来的问题。
    if normalized_name == "ce":  # 如果选择标准交叉熵。
        return CrossEntropySegmentationLoss(ignore_index=ignore_index)  # 返回 CE loss 模块。
    if normalized_name == "dice":  # 如果选择手写 Dice Loss。
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)  # 返回 Dice loss 模块。
    if normalized_name == "ce_dice":  # 如果选择组合损失。
        return CombinedLoss(num_classes=num_classes, ignore_index=ignore_index)  # 返回 CE + Dice loss 模块。
    raise ValueError("loss_name 只能是 ce、dice 或 ce_dice。")  # 如果名称不支持，给出清晰报错。


# 下面这个函数做最小测试：检查三种 loss 都能计算标量并完成反向传播。
def run_minimal_test():  # 定义最小测试函数。
    torch.manual_seed(0)  # 固定随机种子，让测试结果更容易复现。
    batch_size = 2  # 设置 batch 大小为 2。
    num_classes = 8  # 设置类别数为 8，与当前数据集记录保持一致。
    height = 16  # 设置测试图像高度为 16，避免测试太慢。
    width = 20  # 设置测试图像宽度为 20，避免测试太慢。
    ignore_index = -1  # 设置 unknown 标签为 -1。

    targets = torch.randint(0, num_classes, (batch_size, height, width))  # 随机生成 mask，形状是 [B, H, W]。
    targets[0, 0, 0] = ignore_index  # 人为放入一个忽略像素，测试 ignore_index 是否可用。

    print(f"targets shape: {tuple(targets.shape)}")  # 打印 mask 形状，帮助确认标签输入格式。
    print(f"ignore_index in targets: {ignore_index in targets}")  # 打印是否包含 ignore_index。

    for loss_name in ["ce", "dice", "ce_dice"]:  # 依次测试三种损失配置。
        logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)  # 随机生成 logits，形状是 [B, C, H, W]。
        loss_fn = get_loss_function(loss_name, num_classes=num_classes, ignore_index=ignore_index)  # 根据名称创建 loss。
        loss = loss_fn(logits, targets)  # 前向计算 loss。
        loss.backward()  # 反向传播，检查 loss 是否能产生梯度。
        gradient_is_valid = logits.grad is not None and torch.isfinite(logits.grad).all().item()  # 检查梯度是否存在且都是有限值。
        print(f"{loss_name} loss: {loss.item():.6f}")  # 打印当前 loss 的数值。
        print(f"{loss_name} loss shape: {tuple(loss.shape)}")  # 打印 loss 形状，标量应该显示为空元组。
        print(f"{loss_name} gradient valid: {gradient_is_valid}")  # 打印梯度是否正常。
        assert loss.dim() == 0  # 确认 loss 是标量。
        assert gradient_is_valid  # 确认梯度正常。

    print("loss minimal test passed.")  # 如果三种 loss 都通过，打印成功提示。


# 下面这个入口保证直接运行本文件时会执行最小测试。
if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    run_minimal_test()  # 执行最小测试。
