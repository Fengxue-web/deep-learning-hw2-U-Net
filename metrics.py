import torch  # 导入 PyTorch，用于张量计算和语义分割指标统计。
from typing import Dict  # 导入字典类型注解，方便说明函数返回值的结构。


# 下面这个函数负责把模型输出统一转换成类别预测图，方便后续指标计算。
def _to_label_predictions(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # 定义内部辅助函数，输入可以是 logits 或已经 argmax 后的预测标签。
    # 如果 outputs 比 targets 多一个维度，通常说明 outputs 是形状为 [B, C, H, W] 的 logits。
    if outputs.ndim == targets.ndim + 1:  # 判断 outputs 是否包含类别通道维度 C。
        return outputs.argmax(dim=1).long()  # 沿类别维度取最大值下标，得到形状为 [B, H, W] 的预测类别。

    # 如果 outputs 和 targets 维度相同，通常说明 outputs 已经是形状为 [B, H, W] 的预测标签。
    if outputs.ndim == targets.ndim:  # 判断 outputs 是否已经和 targets 具有相同维度。
        return outputs.long()  # 转成 long 类型，保证可以和 mask 标签安全比较。

    # 如果维度既不是 logits 形式，也不是标签形式，就说明输入形状不符合当前函数约定。
    raise ValueError("outputs 应该是 [B, C, H, W] logits 或 [B, H, W] 预测标签。")  # 抛出清晰错误，帮助定位输入形状问题。


# 下面这个函数负责检查预测图和真实标签图的形状是否完全一致。
def _check_same_shape(predictions: torch.Tensor, targets: torch.Tensor) -> None:  # 定义内部辅助函数，用来提前发现 shape 错误。
    # 语义分割指标要求每个像素都有一个预测类别和一个真实类别，所以二者形状必须一样。
    if predictions.shape != targets.shape:  # 如果预测标签和真实标签的形状不同，就无法逐像素比较。
        raise ValueError(f"predictions shape {tuple(predictions.shape)} 与 targets shape {tuple(targets.shape)} 不一致。")  # 抛出包含具体形状的错误。


# 下面这个函数计算 pixel accuracy，也就是所有有效像素中预测正确的比例。
def pixel_accuracy(outputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> float:  # 定义像素准确率函数，默认忽略 unknown 标签 -1。
    # 指标计算不需要梯度，所以使用 no_grad 可以节省显存并避免干扰训练图。
    with torch.no_grad():  # 关闭梯度记录，因为评价指标不参与反向传播。
        predictions = _to_label_predictions(outputs, targets)  # 把 logits 或预测标签统一转换成 [B, H, W] 的预测类别。
        targets = targets.long()  # 把真实标签转成 long 类型，和预测类别保持一致。
        _check_same_shape(predictions, targets)  # 检查预测类别和真实标签是否能逐像素比较。
        valid_mask = targets != ignore_index  # 找出不等于 ignore_index 的有效像素位置。
        valid_count = valid_mask.sum()  # 统计有效像素总数。

        # 如果当前 batch 没有任何有效像素，则返回 0，避免除以 0。
        if valid_count.item() == 0:  # 判断有效像素数量是否为 0。
            return 0.0  # 没有有效像素时返回 0.0，表示无法得到有效准确率。

        correct_count = ((predictions == targets) & valid_mask).sum()  # 统计有效像素中预测正确的数量。
        accuracy = correct_count.float() / valid_count.float()  # 用正确像素数除以有效像素数，得到 pixel accuracy。
        return float(accuracy.item())  # 转成普通 Python float，方便打印、记录日志和写报告。


# 下面这个函数计算 mean IoU，也就是各个出现过的类别 IoU 的平均值。
def mean_iou(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = -1) -> float:  # 定义 mIoU 函数，需要显式传入类别数。
    # 指标计算不需要梯度，所以使用 no_grad 可以让验证过程更轻量。
    with torch.no_grad():  # 关闭梯度记录，因为 mIoU 只用于评价。
        predictions = _to_label_predictions(outputs, targets)  # 把模型输出统一转换成 [B, H, W] 的类别预测。
        targets = targets.long()  # 把真实 mask 转成 long 类型，便于和类别编号比较。
        _check_same_shape(predictions, targets)  # 检查预测和标签的空间形状是否一致。
        valid_mask = targets != ignore_index  # 构造有效像素 mask，用来忽略 unknown 像素。
        iou_values = []  # 创建列表，用来保存每个有效类别的 IoU。

        # 逐个类别计算交集和并集，最后再取平均。
        for class_id in range(num_classes):  # 遍历类别编号，例如 Stanford Background 当前是 0 到 7。
            prediction_is_class = (predictions == class_id) & valid_mask  # 找出预测为当前类别且不是 ignore 的像素。
            target_is_class = (targets == class_id) & valid_mask  # 找出真实标签为当前类别且不是 ignore 的像素。
            intersection = (prediction_is_class & target_is_class).sum().float()  # 交集是预测和真实都属于当前类别的像素数。
            union = (prediction_is_class | target_is_class).sum().float()  # 并集是预测或真实属于当前类别的像素数。

            # 如果某个类别在预测和标签中都没有出现，则跳过该类别，避免无意义的 0/0。
            if union.item() > 0:  # 只有并集大于 0 时，当前类别的 IoU 才有定义。
                iou_values.append(intersection / union)  # 保存当前类别 IoU，数值范围通常在 0 到 1 之间。

        # 如果没有任何类别有有效并集，则返回 0，避免对空列表求平均。
        if len(iou_values) == 0:  # 判断是否没有可计算的类别。
            return 0.0  # 没有有效类别时返回 0.0。

        mean_value = torch.stack(iou_values).mean()  # 把所有类别 IoU 堆叠起来并求平均，得到 mIoU。
        return float(mean_value.item())  # 转成普通 Python float，方便后续日志记录。


# 下面这个类用于在验证集多个 batch 上累计 pixel accuracy 和 mIoU。
class SegmentationMetricsTracker:  # 定义一个简单的指标累计器，后续 train.py 或 evaluate.py 可以复用。
    # 初始化累计器，记录类别数和 ignore_index。
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:  # 定义构造函数，num_classes 是有效类别数量。
        self.num_classes = num_classes  # 保存类别数量，例如当前数据集有效类别数是 8。
        self.ignore_index = ignore_index  # 保存需要忽略的标签值，例如 unknown 标签是 -1。
        self.reset()  # 初始化所有累计统计量。

    # 清空累计结果，通常在每个验证 epoch 开始前调用。
    def reset(self) -> None:  # 定义重置函数，用于重新开始统计。
        self.total_correct = 0.0  # 累计所有 batch 中预测正确的有效像素数。
        self.total_valid = 0.0  # 累计所有 batch 中有效像素总数。
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)  # 为每个类别累计交集像素数。
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)  # 为每个类别累计并集像素数。

    # 用一个 batch 的预测结果和真实标签更新累计统计量。
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:  # 定义更新函数，输入可以是 logits 或预测标签。
        # 指标累计不需要梯度，所以使用 no_grad。
        with torch.no_grad():  # 关闭梯度记录，避免评价指标进入计算图。
            predictions = _to_label_predictions(outputs, targets).detach().cpu()  # 得到预测类别并移动到 CPU，便于累计统计。
            targets_cpu = targets.detach().cpu().long()  # 把真实标签移动到 CPU 并转成 long 类型。
            _check_same_shape(predictions, targets_cpu)  # 检查预测和标签是否具有相同形状。
            valid_mask = targets_cpu != self.ignore_index  # 找出需要参与指标计算的有效像素。
            correct_count = ((predictions == targets_cpu) & valid_mask).sum().item()  # 统计当前 batch 预测正确的有效像素数。
            valid_count = valid_mask.sum().item()  # 统计当前 batch 的有效像素数。
            self.total_correct += float(correct_count)  # 把当前 batch 正确像素数累加到总统计中。
            self.total_valid += float(valid_count)  # 把当前 batch 有效像素数累加到总统计中。

            # 逐个类别累计交集和并集，用于最后计算整个验证集的 mIoU。
            for class_id in range(self.num_classes):  # 遍历所有有效类别编号。
                prediction_is_class = (predictions == class_id) & valid_mask  # 找出预测为当前类别的有效像素。
                target_is_class = (targets_cpu == class_id) & valid_mask  # 找出真实标签为当前类别的有效像素。
                intersection = (prediction_is_class & target_is_class).sum().item()  # 统计当前类别在本 batch 的交集。
                union = (prediction_is_class | target_is_class).sum().item()  # 统计当前类别在本 batch 的并集。
                self.intersections[class_id] += float(intersection)  # 累加当前类别交集。
                self.unions[class_id] += float(union)  # 累加当前类别并集。

    # 根据累计的正确像素数和有效像素数计算整体 pixel accuracy。
    def compute_pixel_accuracy(self) -> float:  # 定义累计版 pixel accuracy 计算函数。
        # 如果还没有有效像素，则返回 0，避免除以 0。
        if self.total_valid == 0:  # 判断累计有效像素是否为 0。
            return 0.0  # 没有有效像素时返回 0.0。

        return self.total_correct / self.total_valid  # 返回整个累计过程中的像素准确率。

    # 根据累计的交集和并集计算整体 mIoU。
    def compute_miou(self) -> float:  # 定义累计版 mIoU 计算函数。
        valid_classes = self.unions > 0  # 只选择在预测或标签中出现过的类别。

        # 如果没有任何类别出现，则返回 0，避免空张量求平均。
        if not bool(valid_classes.any().item()):  # 判断是否没有可计算 IoU 的类别。
            return 0.0  # 没有有效类别时返回 0.0。

        ious = self.intersections[valid_classes] / self.unions[valid_classes]  # 对每个有效类别计算 IoU。
        return float(ious.mean().item())  # 对有效类别 IoU 求平均并转成 Python float。

    # 一次性返回常用指标，方便训练脚本记录日志。
    def compute(self) -> Dict[str, float]:  # 定义统一输出函数，返回一个字典。
        return {  # 返回包含两个核心指标的字典。
            "pixel_accuracy": self.compute_pixel_accuracy(),  # 返回累计 pixel accuracy。
            "miou": self.compute_miou(),  # 返回累计 mIoU。
        }  # 结束字典返回。


# 下面这个函数提供最小测试，不依赖真实数据集，也不会开始训练。
def run_minimal_test() -> None:  # 定义最小测试函数，运行本文件时会调用。
    num_classes = 3  # 构造一个小例子，假设只有 3 个类别：0、1、2。
    ignore_index = -1  # 构造一个 ignored 标签，模拟数据集中的 unknown 像素。
    targets = torch.tensor([[[0, 1, 2], [0, ignore_index, 2]]], dtype=torch.long)  # 创建真实 mask，形状是 [B, H, W]。
    perfect_predictions = torch.tensor([[[0, 1, 2], [0, 0, 2]]], dtype=torch.long)  # 创建完全正确的预测，ignore 像素处随便填 0。
    mixed_predictions = torch.tensor([[[0, 2, 2], [1, 1, 0]]], dtype=torch.long)  # 创建一个有对有错的预测，用来检查指标不是总为 1。
    perfect_logits = torch.nn.functional.one_hot(perfect_predictions, num_classes=num_classes).permute(0, 3, 1, 2).float()  # 把预测标签转成 [B, C, H, W] logits 形式。

    # 分别测试直接输入预测标签和输入 logits 两种情况。
    perfect_acc = pixel_accuracy(perfect_predictions, targets, ignore_index=ignore_index)  # 计算完全正确预测的 pixel accuracy。
    perfect_miou = mean_iou(perfect_predictions, targets, num_classes=num_classes, ignore_index=ignore_index)  # 计算完全正确预测的 mIoU。
    logits_acc = pixel_accuracy(perfect_logits, targets, ignore_index=ignore_index)  # 测试函数能否处理 [B, C, H, W] 形式的 logits。
    mixed_acc = pixel_accuracy(mixed_predictions, targets, ignore_index=ignore_index)  # 计算部分错误预测的 pixel accuracy。
    mixed_miou = mean_iou(mixed_predictions, targets, num_classes=num_classes, ignore_index=ignore_index)  # 计算部分错误预测的 mIoU。

    # 测试累计器是否能在一个 batch 上得到与普通函数一致的结果。
    tracker = SegmentationMetricsTracker(num_classes=num_classes, ignore_index=ignore_index)  # 创建语义分割指标累计器。
    tracker.update(perfect_predictions, targets)  # 用完全正确预测更新一次累计器。
    tracker_result = tracker.compute()  # 计算累计器中的 pixel accuracy 和 mIoU。

    # 打印关键结果，方便确认每一步的含义。
    print(f"targets shape: {tuple(targets.shape)}")  # 打印真实标签形状，预期是 [1, 2, 3]。
    print(f"perfect pixel accuracy: {perfect_acc:.4f}")  # 打印完全正确预测的像素准确率，预期接近 1。
    print(f"perfect mIoU: {perfect_miou:.4f}")  # 打印完全正确预测的 mIoU，预期接近 1。
    print(f"logits pixel accuracy: {logits_acc:.4f}")  # 打印 logits 输入形式下的像素准确率，预期接近 1。
    print(f"mixed pixel accuracy: {mixed_acc:.4f}")  # 打印部分错误预测的像素准确率，预期小于 1。
    print(f"mixed mIoU: {mixed_miou:.4f}")  # 打印部分错误预测的 mIoU，预期小于 1。
    print(f"tracker result: {tracker_result}")  # 打印累计器结果，预期两个指标都接近 1。

    # 使用断言检查核心预期，任何一个不满足都会直接报错。
    assert abs(perfect_acc - 1.0) < 1e-6  # 完全正确预测的 pixel accuracy 应该等于 1。
    assert abs(perfect_miou - 1.0) < 1e-6  # 完全正确预测的 mIoU 应该等于 1。
    assert abs(logits_acc - 1.0) < 1e-6  # logits 输入形式下的 pixel accuracy 也应该等于 1。
    assert abs(tracker_result["pixel_accuracy"] - 1.0) < 1e-6  # 累计器的 pixel accuracy 应该等于 1。
    assert abs(tracker_result["miou"] - 1.0) < 1e-6  # 累计器的 mIoU 应该等于 1。
    assert 0.0 <= mixed_acc <= 1.0  # 部分错误预测的 pixel accuracy 应该落在 0 到 1 之间。
    assert 0.0 <= mixed_miou <= 1.0  # 部分错误预测的 mIoU 应该落在 0 到 1 之间。
    print("metrics minimal test passed.")  # 打印测试通过信息。


# 下面这个入口保证直接运行 python src/metrics.py 时会执行最小测试。
if __name__ == "__main__":  # 判断当前文件是否作为脚本直接运行。
    run_minimal_test()  # 运行最小测试函数。
