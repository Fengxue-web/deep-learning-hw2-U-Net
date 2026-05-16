# 本文件定义语义分割任务使用的 U-Net 模型。

# 导入 PyTorch 主包，方便测试代码创建 tensor。  # 这里会用到 torch.randn 和 torch.no_grad。
import torch  # PyTorch 的 tensor 库。

# 导入 PyTorch 的神经网络层模块。  # 这里会用到 nn.Module、Conv2d、MaxPool2d 等。
import torch.nn as nn  # torch.nn 的简写名称。

# 导入 PyTorch 的函数式工具。  # 这里只在需要对齐特征图尺寸时使用。
import torch.nn.functional as F  # torch.nn.functional 的简写名称。


# 这个类实现 U-Net 中最基础的双卷积模块。  # 每个模块都会保持输入的高和宽不变。
class DoubleConv(nn.Module):  # 该模块结构为 Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU。
    # 这个函数创建一个双卷积模块内部需要的所有层。  # in_channels 是输入通道数，out_channels 是输出通道数。
    def __init__(self, in_channels, out_channels):  # 创建一个双卷积模块。
        super().__init__()  # 初始化父类 nn.Module。
        self.layers = nn.Sequential(  # 按实际执行顺序保存这些网络层。
            nn.Conv2d(  # 创建第一个 3x3 卷积层。
                in_channels,  # 输入通道数，例如 RGB 图像为 3。
                out_channels,  # 输出特征通道数。
                kernel_size=3,  # 使用 3x3 卷积核。
                padding=1,  # 让本次卷积后的 H 和 W 保持不变。
                bias=False,  # 后面接了 BatchNorm，因此卷积层不需要 bias。
            ),  # 第一个卷积层定义结束。
            nn.BatchNorm2d(out_channels),  # 对输出通道做归一化，让训练更稳定。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活，并尽量原地修改以节省内存。
            nn.Conv2d(  # 创建第二个 3x3 卷积层。
                out_channels,  # 第二个卷积层的输入通道数等于第一个卷积层的输出通道数。
                out_channels,  # 在这个模块内部保持输出通道数不变。
                kernel_size=3,  # 再次使用 3x3 卷积核。
                padding=1,  # 再次保持 H 和 W 不变。
                bias=False,  # 后面接了 BatchNorm，因此卷积层不需要 bias。
            ),  # 第二个卷积层定义结束。
            nn.BatchNorm2d(out_channels),  # 对第二个卷积层的输出做归一化。
            nn.ReLU(inplace=True),  # 使用第二次 ReLU 激活。
        )  # 顺序层列表定义结束。

    # 这个函数描述输入数据如何通过该模块。  # x 的形状是 [B, in_channels, H, W]。
    def forward(self, x):  # 执行双卷积模块的前向传播。
        return self.layers(x)  # 返回形状为 [B, out_channels, H, W] 的 tensor。


# 这个类实现编码器中的一次下采样。  # 它先把 H 和 W 减半，然后再做双卷积。
class DownBlock(nn.Module):  # 该模块结构为 MaxPool2d -> DoubleConv。
    # 这个函数创建下采样模块。  # in_channels 来自前一个编码器阶段的输出通道数。
    def __init__(self, in_channels, out_channels):  # 创建一个下采样模块。
        super().__init__()  # 初始化父类 nn.Module。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 将特征图的高和宽都缩小为原来的一半。
        self.conv = DoubleConv(in_channels, out_channels)  # 在下采样之后继续学习更深层特征。

    # 这个函数执行一次下采样过程。  # x 的形状是 [B, in_channels, H, W]。
    def forward(self, x):  # 执行最大池化和双卷积。
        x = self.pool(x)  # 将 [B, C, H, W] 转换为 [B, C, H/2, W/2]。
        x = self.conv(x)  # 将通道数从 in_channels 转换为 out_channels。
        return x  # 返回下采样后的特征图。


# 这个类实现解码器中的一次上采样。  # 它会上采样、拼接 skip 特征，然后再做双卷积。
class UpBlock(nn.Module):  # 该模块结构为 ConvTranspose2d -> 拼接 skip connection -> DoubleConv。
    # 这个函数创建上采样模块。  # in_channels 来自更深层特征，skip_channels 来自编码器的跳跃连接特征。
    def __init__(self, in_channels, skip_channels, out_channels):  # 创建一个上采样模块。
        super().__init__()  # 初始化父类 nn.Module。
        self.up = nn.ConvTranspose2d(  # 创建可学习的上采样层。
            in_channels,  # 更深层特征图的通道数。
            out_channels,  # 上采样后的通道数。
            kernel_size=2,  # 使用 2x2 转置卷积核。
            stride=2,  # 将高和宽都放大为原来的 2 倍。
        )  # 转置卷积层定义结束。
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)  # 融合上采样特征和 skip 特征。

    # 这个函数执行一次上采样过程。  # x 是更深层特征图，skip 是编码器传来的特征图。
    def forward(self, x, skip):  # 执行上采样、skip 拼接和双卷积。
        x = self.up(x)  # 将 H 和 W 放大 2 倍，同时把通道数变为 out_channels。
        if x.shape[-2:] != skip.shape[-2:]:  # 检查 x 的 H 和 W 是否与 skip 特征图一致。
            x = F.interpolate(  # 只有在尺寸有轻微不一致时才调整 x 的大小。
                x,  # 需要调整尺寸的特征图。
                size=skip.shape[-2:],  # 目标高和宽来自 skip 特征图。
                mode="bilinear",  # 对特征图使用双线性插值。
                align_corners=False,  # 双线性插值中常用且稳定的设置。
            )  # 尺寸调整操作结束。
        x = torch.cat([skip, x], dim=1)  # 沿通道维度 C 拼接 skip 特征和上采样特征。
        x = self.conv(x)  # 用双卷积融合拼接后的特征图。
        return x  # 返回解码后的特征图。


# 这个类实现完整的 U-Net 模型。  # 输入形状是 [B, 3, H, W]，输出形状是 [B, num_classes, H, W]。
class UNet(nn.Module):  # 这是一个带 skip connection 的经典编码器-解码器 U-Net。
    # 这个函数创建所有编码器、瓶颈层、解码器和输出层。  # 当前数据集的 num_classes 为 8。
    def __init__(self, in_channels=3, num_classes=8, features=(64, 128, 256, 512)):  # 创建完整的 U-Net 模型。
        super().__init__()  # 初始化父类 nn.Module。
        self.encoder1 = DoubleConv(in_channels, features[0])  # 第一个编码器阶段保持图像尺寸不变。
        self.encoder2 = DownBlock(features[0], features[1])  # 第二个编码器阶段使用 1/2 图像尺寸。
        self.encoder3 = DownBlock(features[1], features[2])  # 第三个编码器阶段使用 1/4 图像尺寸。
        self.encoder4 = DownBlock(features[2], features[3])  # 第四个编码器阶段使用 1/8 图像尺寸。
        self.bottleneck = DownBlock(features[3], features[3] * 2)  # 最底部瓶颈层使用 1/16 图像尺寸。
        self.decoder4 = UpBlock(features[3] * 2, features[3], features[3])  # 从 1/16 尺寸解码回 1/8 尺寸。
        self.decoder3 = UpBlock(features[3], features[2], features[2])  # 从 1/8 尺寸解码回 1/4 尺寸。
        self.decoder2 = UpBlock(features[2], features[1], features[1])  # 从 1/4 尺寸解码回 1/2 尺寸。
        self.decoder1 = UpBlock(features[1], features[0], features[0])  # 从 1/2 尺寸解码回原始尺寸。
        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)  # 将特征图映射为每个类别的 logits。

    # 这个函数定义完整的前向传播过程。  # x 的形状是 [B, 3, H, W]。
    def forward(self, x):  # 将输入图像 tensor 通过 U-Net 转换为输出 logits。
        enc1 = self.encoder1(x)  # enc1 的形状是 [B, 64, H, W]。
        enc2 = self.encoder2(enc1)  # enc2 的形状是 [B, 128, H/2, W/2]。
        enc3 = self.encoder3(enc2)  # enc3 的形状是 [B, 256, H/4, W/4]。
        enc4 = self.encoder4(enc3)  # enc4 的形状是 [B, 512, H/8, W/8]。
        bottleneck = self.bottleneck(enc4)  # bottleneck 的形状是 [B, 1024, H/16, W/16]。
        dec4 = self.decoder4(bottleneck, enc4)  # 在这个解码阶段使用 enc4 作为 skip connection。
        dec3 = self.decoder3(dec4, enc3)  # 在这个解码阶段使用 enc3 作为 skip connection。
        dec2 = self.decoder2(dec3, enc2)  # 在这个解码阶段使用 enc2 作为 skip connection。
        dec1 = self.decoder1(dec2, enc1)  # 在这个解码阶段使用 enc1 作为 skip connection。
        logits = self.classifier(dec1)  # logits 的形状是 [B, num_classes, H, W]。
        return logits  # 返回未经过 softmax 的 logits，供 CrossEntropyLoss 或后续 Dice Loss 使用。


# 这个函数运行一个不依赖数据集的极小形状测试。  # 它只检查模型前向传播是否能正常工作。
def run_shape_test():  # 运行本文件的最小测试。
    torch.manual_seed(0)  # 固定随机种子，让随机输入可以复现。
    batch_size = 2  # 使用两张假图像组成一个 batch。
    image_channels = 3  # 使用 RGB 输入图像。
    num_classes = 8  # 使用 PROJECT_STATUS.md 中记录的 Stanford Background 有效类别数。
    height = 240  # 使用 dataset 测试中记录的图像高度。
    width = 320  # 使用 dataset 测试中记录的图像宽度。
    model = UNet(in_channels=image_channels, num_classes=num_classes)  # 创建 U-Net 模型。
    model.eval()  # 切换到评估模式，用于这个简单的前向传播测试。
    x = torch.randn(batch_size, image_channels, height, width)  # 创建形状为 [B, 3, H, W] 的假输入 tensor。
    with torch.no_grad():  # 关闭梯度追踪，因为这里不是训练。
        logits = model(x)  # 执行模型前向传播，得到形状为 [B, num_classes, H, W] 的 logits。
    expected_shape = (batch_size, num_classes, height, width)  # 定义模型应该输出的形状。
    print(f"input shape: {tuple(x.shape)}")  # 打印输入 tensor 的形状。
    print(f"logits shape: {tuple(logits.shape)}")  # 打印输出 logits tensor 的形状。
    print(f"expected logits shape: {expected_shape}")  # 打印期望的输出形状。
    assert tuple(logits.shape) == expected_shape  # 如果输出形状不正确，就停止程序并报错。
    print("U-Net shape test passed.")  # 当形状正确时，打印测试通过信息。


# 这个代码块让我们可以从命令行直接运行本文件。  # 命令是 python src/model_unet.py。
if __name__ == "__main__":  # 只有直接执行本文件时，才运行下面的测试。
    run_shape_test()  # 运行最小形状测试。
