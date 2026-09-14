传统滤波器是在手工规定估计器的函数族；神经网络则通过数据学习估计器。它的优势不只是“更强的非线性”，而是把大量离线计算、场景统计规律和时空先验，摊销成一次实时前向推理。
因此，你最合适的入口不是泛泛地学“人工智能”，而是从下面这个等式开始：
\[
f^*(X)=\arg\min_f \mathbb E[\|f(X)-Y\|_2^2]
      =\mathbb E[Y\mid X]
\]其中：
- \(X\)：低 spp radiance、normal、depth、roughness、motion、hitT 等可观测量
- \(Y\)：无限采样或高 spp 的真实图像
- \(f_\theta\)：神经网络
- 用 MSE 训练时，网络逼近的是给定观测后的条件均值
这可以把你已有的全部经验接到深度学习上。
先修正“预测 kernel weight”这个理解
传统 denoiser 通常写成：
\[
\hat L_p=\sum_{q\in N(p)}w_{pq}(X)L_q
\]你尝试用 Monte Carlo 重建透明通道的 \(w_{pq}\)，而 KPCN 做的事情正是：
\[
w_{pq}=g_\theta(X_p,X_q)
\]即让 CNN 预测空间变化的滤波 kernel。2017 年的 KPCN 应该成为你的第一篇神经渲染论文，它和你目前的思想几乎是直接衔接的。
但现代神经降噪器不一定显式输出 kernel，还可能：
- 直接预测 clean radiance
- 预测 noisy radiance 的 residual
- 在隐空间里滤波，再解码
- 预测可分离或低秩 kernel
- 用 recurrent latent state 表示跨帧历史
- 同时进行重投影、抗锯齿、降噪和超分
显式 kernel prediction 可解释、保能量，但存在两个限制：感受野内没有出现的值无法生成；输出 \(K^2\) 个权重的带宽和计算量很大。这正好对应你碰到的问题。
第一阶段：只学必要的神经网络基础
建议用两到三周，只掌握这些概念：

- 张量和矩阵乘法
- 链式法则与反向传播
- computation graph 和 automatic differentiation
- SGD、Adam、learning rate
- training / validation / test 的区别
- 过拟合、正则化与泛化
- convolution、receptive field、downsample/upsample
- residual connection、U-Net
- MSE、L1、Charbonnier、相对误差和 HDR 压缩 loss
不要从 Transformer、LLM、GAN 或 diffusion 开始，它们会把你带离当前问题。
实践顺序：
1. 跟着 PyTorch 官方基础教程 完成 tensor、autograd、model、optimizer、dataloader。
2. 阅读并亲手复现 micrograd，理解反向模式自动微分。
3. 选择性阅读《Deep Learning》第 2–6、8、9、11 章；不需要从头到尾啃。
4. 学习 CS231n 的 CNN 部分。
以你的图形学背景，真正需要补的是反向传播、优化和泛化，而不是重新学习向量矩阵运算。
第二阶段：用 SuzuRenderer 做第一个训练实验
不要先做透明或时域。先训练一个单帧、分通道的离线原型。
数据输入建议直接复用你已有的 M1 ABI：
- 1 spp demodulated diffuse radiance
- 1 spp demodulated specular radiance
- diffuse/specular factor
- normal
- linear viewZ
- roughness
- diffuse/specular hitT
- material/domain tag
目标使用独立随机种子的 512–4096 spp reference。
仓库现有的输入定义已经可以作为数据集 schema：
[denoiser-m1-input-contract.md (line 1)](F:/WorkSpace/SuzuRenderer/SuzuRenderer/docs/denoiser-m1-input-contract.md:1)
先做三个递进模型：
1. 5×5 softmax kernel prediction
   验证“学习 kernel weight”能达到什么程度。
2. 可分离 kernel prediction
   输出水平和垂直 kernel，把 \(K^2\) 成本降到 \(2K\)。
3. 小型 residual U-Net
   直接预测 clean radiance 或 residual，与 kernel prediction 做 A/B。
这三个实验会非常直观地回答：性能差异来自 kernel 表达能力、感受野，还是直接预测本身。
数据比网络结构更重要
训练集必须满足：
- noisy input 与 reference 使用独立随机种子
- 按“场景”划分 train/validation/test，而不是随机拆相邻帧
- 包含不同材质、光照、曝光、roughness、相机距离和运动
- 保留 scene-linear HDR 数据
- 同时报告线性域和压缩域误差
- 单独统计高光、边缘、失遮挡、玻璃和低概率高能路径
HDR 可以先用
\[
T(L)=\operatorname{asinh}(sL)
\]或者 log1p 压缩后计算 Charbonnier/L1，再叠加一个权重较小的线性域相对误差。只在 tonemap 后训练会让模型倾向于视觉好看，而不一定能量正确。
Intel OIDN 的公开仓库包含完整 Python 训练工具、数据预处理和自定义权重导出流程，很适合作为工程参考；它明确支持用 noisy/reference/AOV 数据训练自有模型。OIDN training toolkit
第三阶段：再进入时域神经降噪
单帧模型稳定后，再训练序列模型。推荐从 2017 年 NVIDIA 的 Recurrent Denoising Autoencoder 开始。
第一版模型不需要复杂：
当前帧 noisy signals
+ 当前 guides
+ 重投影后的上一帧输出
+ reprojection validity
        ↓
small U-Net
        ↓
当前 clean estimate
第二版再把上一帧 RGB history 换成 ConvGRU 或 recurrent latent feature。
训练序列应包含：
- 普通相机运动
- 刚体运动
- 遮挡与失遮挡
- 光照变化
- camera cut
- frame gap
- resize/reset
- 每帧独立的随机采样序列
Loss 可以由当前帧重建误差和有效重投影区域的时域一致性组成，但不能在失遮挡区域强行施加 temporal loss。
最后才处理透明路径
神经网络不能从不存在的信息中恢复唯一答案。假如两个不同的透明路径在你提供的 \(X\) 上完全相同，网络只能输出它们的条件平均，这会表现为模糊、能量偏差或幻觉。
所以透明通道的突破仍然需要表示设计：
- primary surface replacement
- secondary surface position/normal/depth
- transmission hitT
- path-domain tag
- transmittance
- refracted virtual motion
- path-space layer decomposition
Learning 可以替代复杂的手工映射，但不能替代必要的可观测量。DLSS-RR 的优势来自模型、训练数据、时域系统、guide 设计和部署优化的共同作用，不只是“网络更深”。
最适合你的近期目标是：
用 SuzuRenderer 生成自己的 paired dataset，在 PyTorch 中训练一个分 diffuse/specular 的 KPCN 和 residual U-Net，并与现有 RELAX/MCRR 做固定场景 A/B。

完成这个项目后，你对深度学习的理解会比完成一整套通用分类课程更扎实，也会自然知道下一步该走 temporal reconstruction、transparent path representation，还是实时推理优化。