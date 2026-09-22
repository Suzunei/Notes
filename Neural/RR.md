## 神经网络降噪论文调研

reference:

Temporally Stable Real-Time Joint Neural Denoising and Supersampling 2022 Intel

Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network 2021 CAD&CG

Streaming-Aware Neural Monte Carlo Rendering Framework with Unified Denoising-Compression and Client Collaboration 2025 CAD&CG

Joint Denoising and Upscaling via Multi-branch and Multi-scale Feature Network 2025 AMD



### 1.Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network

传统方法

PT——Denoiser——TAA——Upscaler

问题：

Denoiser与Upscaler为**至少**两个网络（denoiser可能会分为diffuse/specular两个不同网络），成本太高

这个方法使用g-buffer提取公共的Feature，并且通过U-Net通过改变分辨率得到cascade的结果（效果类似于A-trous）获得更大感受野（解决kernel过大的问题HxWx$K^2$）.

最终用公共的Feature训练由4个kernel（diffuse filter/specular filter/albedo filter/upscaling/filter）组成的multi-task neural rr network用于预测Kernels' weight K($w_i$)和TAA的blend weight

这个weight本身是通过g-buffer中的几何信息人为拟合的，是一系列weight的乘积，这里用一个网络表示

也就是这个方法本身起作用的部分是传统的

流程可以总结为：

```
                      Frame t
                        │
          ┌─────────────┴─────────────┐
          │                           │
      Diffuse_t                   Specular_t
          │                           │
   Albedo demod.                 + Roughness
          │                           │
          └─────────────┬─────────────┘
                        │
                 + Albedo / Normal
                 + Motion Vector
                        │
                        │
Previous filtered outputs t-1
          │
          ▼
    Motion-vector Warp
          │
          ▼
   Warped History t-1
          │
          ├─────────────────────────┐
          │                         │
          ▼                         ▼
     Current Frame             History Frame
          │                         │
          │     Quantized U-Net     │
          └───────► Features ◄──────┘
                        │
                        ▼
              Predict per-pixel:
              ┌──────────────────┐
              │ K_current  3×3   │
              │ K_history  3×3   │
              │ α_temporal       │
              └──────────────────┘
                        │
                        ▼
               Spatiotemporal Filter
                        │
        ┌───────────────┴────────────────┐
        │                                │
 K_current * Current             K_history * History
        │                                │
        └───────────────┬────────────────┘
                        │
                  temporal blend
                        │
                        ▼
                temporally accumulated
                       signal
                        │
                        ▼
             hierarchical spatial filter
                        │
              ┌─────────┴────────┐
              ▼                  ▼
          Diffuse              Specular
              │                  │
              └────────┬─────────┘
                       ▼
                   Composite
                       │
                  Upsampling
```

cascade predicted weight K:

```
U-Net Feature Extractor

1/8 scale feature
      │
      └── predict K_1/8
      │
      ▼
1/4 scale feature
      │
      └── predict K_1/4
      │
      ▼
1/2 scale feature
      │
      └── predict K_1/2
      │
      ▼
native-resolution Output Block
      │
      ├── predict K_current 3×3
      ├── predict K_history 3×3
      ├── predict temporal blend α
      └── predict auxiliary kernels
```

A-trous Filter:

```
Current / History
      │
spatiotemporal filter   ← K_native
      │
      ▼
downsample
      │
filter                  ← K_1/2
      │
downsample
      │
filter                  ← K_1/4
      │
...
      │
upsample + skip
      │
filter                  ← 对应尺度 kernel
      ▼
result
```

**存在哪些问题：**

**1.目前看来主要是对于specular的网络的训练问题，没有考虑hitT/roughness等参数训练出来的specular表面的效果肯定是不够的**

**2.网络大小为:W x H x $K_t^3$**，虽然复杂度相较于W x H x $K^2$有所降低，但是复杂度依然多出来一个维度

### 2.Real-time Monte Carlo Denoising with Weight Sharing Kernel Prediction Network CAD&CG

传统的Kernel prediction的做法是：
$$
CNN\rightarrow K_p
$$
对于每个Pixel p ，直接预测：
$$
K_p\in\mathbb R^{k\times k}
$$
如果kernel 是13 x 13,那么最后一层对每个pixel要输出:
$$
13^2 = 169
$$
个值。

所以输出的tensor是：
$$
H\times W\times169
$$
而这个超宽的输出层就会严重拖慢网络的实时效率，这也是移动端使用网络的瓶颈之一。

论文中提出了一个方法，我们对于每个pixel不去预测kernel,取而代之的是预测一个scalar

也就是值预测一个:
$$
I(p)
$$
一个标量。

所以整张图就是:
$$
I\in\mathbb R^{H\times W\times1}
$$
也就是预测一个importance map

最终对这些值做softmax:

$$
w_p(q)
=
\frac{\exp I(q)}
{\sum_{q'\in\Omega_p}\exp I(q')}
$$

输出一个kernel:

$$
K_p=
\begin{bmatrix}
w_1&w_2&w_3\\
w_4&w_5&w_6\\
w_7&w_8&w_9
\end{bmatrix}
$$

因此网络的大小从原来的：

$$
H\times W\times k^2
$$
缩成了:

$$
H\times W\times1
$$

**这样就能得到比U-Net的级联处理更小的网络，损失了精度但是很大程度的降低了workload，适合移动端。**

**目前并没有人使用这个思路进行优化**

### 3.Streaming-Aware Neural Monte Carlo Rendering Framework with Unified Denoising-Compression and Client Collaboration 2025 CAD&CG

**对应reconstruct color.**

1.核心设计：

对端云路径追踪的优化，客户端只需要提供g-buffer的几何/材质信息，把路径追踪需要的:

$$
\text{Adaptive Sampling}
+
\text{Denoising}
+
\text{Compression}
$$

做成一个联合优化的neural framwork，再根据客户端自身g-buffer中的材质/几何信息进行解码，得到output frame color

也就是把传统方法中的denoiser/compression/full screen pt +vrrt (about 4 networks)变成了一个network联合训练adaptive sampling/denoising filter reconstruction/compression

最后云上会训练好一个压缩过的latent流式传给端侧，端侧会做两件事，第一是运行一个非常轻量的神经网络来得到g-buffer的feature，第二是将这个feature作为云侧传入的latent的offset，将latent经过端侧修正的结果传入decoder，最终在端侧得到一个out put frame color.

大概流程如下：

```
Server
────────────────────────────
Sampling Estimation Network
        │
        ▼
Adaptive Sampling
        │
        ▼
Path Tracing
        │
        ▼
Temporal Accumulation
        │
        ▼
Unified Denoising-Compression
Encoder E
        │
        ▼
Quantized latent x̂
        │
   Entropy encode
        │
        │ bitstream
════════╪════════ network
        │
        ▼
   Entropy decode
        │
        ▼
     latent x̂
        │
        │
        ├───────────── Client G-buffer
        │                    │
        │                    ▼
        │              G-buffer Encoder Eg
        │                    │
        │                   f_t
        │                    │
        ▼                    ▼
              Decoder D
                 │
                 ▼
             Final frame


Client(detail)
────────────────────────────
G-buffer rendering
G-buffer Encoder Eg
QR offset predictor
Decoder D

                 Network
                    │
              bitstream b_t
                    │
             Entropy Decode
                    │
                    ▼
             latent x̂_t
                    │
                    │
                    ├───────────────┐
                    │               │
                    │               │
Local scene         │               │
    │               │               │
Raster G-buffer     │               │
    │               │               │
    ▼               │               │
   g_t              │               │
    │               │               │
    ▼               │               │
G-buffer Encoder E_g│               │
    │               │               │
    ▼               │               │
   f_t              │               │
    │                │              │
    ├──► T(f_t) ────┘              │
    │         │                     │
    │     x̂_t + offset             │
    │         │                     │
    └─────────┼─────────────────────┘
              ▼
        Neural Decoder D
              │
              ▼
       final frame x̃_t
```



### 4.Joint Denoising and Upscaling via Multi-branch and Multi-scale Feature Network AMD 2025

在2022的方法基础上的结构优化版本

最核心的改进：

1.将Radiance和G-buffer分为两个branch

因为两个输入信号的性质其实完全不同：

对于1spp SceneColor，存在严重的Monte carlo noise:

```
signal + Monte Carlo noise
```

但是对于G-buffer来说它基本上没有MC noise，但是存在走样

因此2022的方法把：

$$
[Color,Normal,Albedo,Roughness]
\rightarrow U-Net
$$

全部concat到一起，并不能得到最合理的feature extraction

因此改成：

```
                 MUNet

       Radiance Branch     Guiding Branch

Noisy Color ───────►        ◄────── G-buffer
History Radiance            History G-buffer
       │                         │
       ▼                         ▼
     feature                   feature
       │                         │
       │◄──── modulation ────────┘
       ▼
 Radiance feature
```

同时Guiding branch可以很小，因为它没有MC noise,并且信息冗余很大，因此：

$$
C_{guide}<C_{radiance}
$$


guiding branch 更浅，channel更少

这也是lightweight MUNet 的关键设计之一

从移动端的角度，这个也有可借鉴之处：不要对color 和g-buffer花费相同的neural compute budget



2.对U-Net layer做了 partial Convolution

除了第一层encoder和最后一层decoder其他层都不对全部的channel做convolution



3.implicit radiance decomposition：不再显示分解diffuse 和 specular ，只需要scene color

然后直接构造：

$$
X=
\left[
\frac{\bar I_r}{\bar A_d},
\frac{\bar I_r}{\bar A_s},
\bar I_r
\right]
$$

分别为：SceneColor/DiffuseAlbedo/SpecularAlbedo

然后进行denoising：
$$
o_d=D(X)
$$


这里会得到一个9 channels result，可以看做三个通道的RGB 重建结果

然后再进行composition weights prediction，得到：

$$
\rho,\sigma,\tau
$$

进行重组:

$$
o_t
=
\rho\,o_d[0:3]\bar A_d
+
\sigma\,o_d[3:6]\bar A_s
+
\tau\,o_d[6:9]
+
\kappa I_r^h
$$


最后再blend history

$$
O_t
=
lerp(o_t,O_{t-1},\omega)
$$


所以实际上是让网络自己判断，当前pixel被解释成diffuse-like,specular-like or radiance，也就是implicit radiance decomposition

而不要求 rendering pipeline 显示输出: diffuse channel + specular channel + demoduated

其余部分与2022的方法相似

整个流程如下：

                                           Frame t

            1spp noisy SceneColor
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
               Radiance input    G-buffer
                                 Normal
                                 Roughness
                                 Diff/Spec Albedo
                     │              │
            History Radiance    History Guide
                     │              │
                     ▼              ▼
                 ┌──────────────────────┐
                 │ Multi-branch U-Net   │
                 │                      │
                 │ Radiance Branch      │
                 │      ▲               │
                 │      │ modulation    │
                 │ Guiding Branch ──────┘
                 └──────────┬───────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
        Feature 0        Feature 1         Feature 2
           │                │                │
           ▼                ▼                ▼
         3×3 filter       3×3 filter       3×3 filter
           └──────────── multi-scale ─────────┘
                            │
                            ▼
                      Target resolution
                            │
                   Temporal Accumulation
                            │
                            ▼
                   Implicit decomposition

            [ I/A_diff , I/A_spec , I ]
                            │
                            ▼
                    Multi-scale Denoise
                            │
                            ▼
              ρ·Diffuse + σ·Specular
                 + τ·Radiance + κ·Input
                            │
                            ▼
                     History blend ω
                            │
                            ▼
                       Final 4K



**并且这个方法的sr与rr是联合的，直接从1spp 1080p 输出 Denoised 4K**



和2022 JNDS对比：

|                 | 2022 JNDS               | 2025 AMD                           |
| --------------- | ----------------------- | ---------------------------------- |
| 输入            | Diffuse + Specular 分离 | **一个 SceneColor**                |
| G-buffer        | 与 radiance 一起处理    | **单独 Guiding Branch**            |
| Feature Net     | U-Net                   | **Multi-branch U-Net**             |
| Denoising       | hierarchical kernel     | multi-scale Gaussian filter        |
| Diff/Spec       | 显式分解                | **Implicit decomposition**         |
| Temporal        | learned kernel + blend  | learned accumulation/filter weight |
| 超分            | joint                   | joint                              |
| 参数量          | 2.7 M                   | **0.644 M**                        |
| 1080p→4K GFLOPs | 386                     | **396**                            |

**在参数量和降噪效果上都有明显提升**



#### 存在的问题：

但是这篇论文最后有一个比较大的问题：也就是其feature extractor很轻，但是filtering stage很重

其GPU time是JNDS的约 1.8x

**因此这个方法和CAD&CG的WSKP非常互补**



# BVH构建算法调研

## 1. SAH
The Surface Area Heuristic（SAH），表面启发式方法，提供一种量化判断bbox划分结果的方法。
基本思想如下：
建树时，对内部有n个图元，不再向下拆分的情况，我们记光线与图元相交的时间为$t_{isect}(i)$。则对BVH的叶子节点有：

$$
Cost = \sum^n_{i = 1}t_{isect}(i)
$$

当需要继续拆分时，开销为：

$$
Cost(V \rarr L, R) = t_{trav} + p_L \sum^{N_L}_{i = 1}t_{isect}(L_i) + p_R \sum^{N_R}_{i = 1}t_{isect}(R_i)
$$

其中，$L, R$为左右子包围盒；$N_L, N_R$为子包围盒中的图元数量；$p_L, p_R$是光线与子包围盒相交的概率，$t_{trav}$是从根节点计算到当前步骤进行判定的总开销

关键问题在于计算$p_L, p_R$
由 Cauchy–Crofton 公式，有结论：
> 给定一个空间中的凸几何体，如果在空间任意角度任意位置存在直线，那么这条直线穿过这个几何体的概率正比于这个几何体的表面积

由此有：
> 如果空间中一个几何体O包含几何体A，空间中的某一条直线与几何体O相交的情况下，它同时与内部几何体A相交的概率为两个几何体表面积之比

因此有：
$$
p_A = P(A|O) = S_A / S_O
$$

通常，我们假设所有图元求交时间相同。于是公式简化为：

$$
Cost(V \rarr {L, R}) = K_T + K_I(\frac{SA(V_L)}{SA(V)} N_L + \frac{SA(V_R)}{SA(V)} N_R)
$$

$K_T$即$t_{trav}$，$K_I$为统一设置的求交时间值

SAH方法比较原始，GPU并行效率不高（因为是二叉树，而且下层对上层有依赖关系，上层节点少图元多，还并行不了几个线程）

## 2. SBVH

离线渲染，暂时不考虑

## 3. LBVH

Linear BVH（LBVH）的基本思想是，不必要花费大量时间找到最优的划分，只需要把空间中靠得近的放在一起建树即可。我们期望空间中图元的编码本身就能反映他们的空间关系，这样我们就能直接排序并做分割。

### 1. Morton Code

现在，基本问题转化成：如何将图元的空间坐标转化为可排序的ID，使得空间上接近的点在数值上也相近。为此我们引入了Morton Code，莫顿码。莫顿码在排序后也被称为 z-order curve/z-order code

莫顿码的构建方式是将坐标的二进制bit交错排列。比如，对
```text
x = xn ... x2 x1 x0
y = yn ... y2 y1 y0
z = zn ... z2 z1 z0

-> xn yn zn ... x2 y2 z2 x1 y1 z1 x0 y0 z0
```

莫顿码的作用原理如下：
考虑一个4x4的网格，将其编码：

```text
(0,0): 0000 (1,0): 0001 (2,0): 0100 (2,1): 0101
(0,1): 0010 (1,1): 0011 (2,1): 0110 (3,1): 0111
(0,2): 1000 (1,2): 1001 (2,2): 1100 (2,3): 1101
(0,3): 1010 (1,3): 1011 (2,3): 1110 (3,3): 1111
```
我们发现这样编码可以很容易地通过每一位的值来判断所属的空间位置关系。越相近的图元，其高位相同的位数越多；相同的位即表示两者所属某个同一粒度的空间。

基于此我们得到了BVH的基本建立思路：
1. 为图元分配莫顿码
2. 对莫顿码排序
3. 逐位检查莫顿码不同的位置，进行分割

示意图见：https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/  **Top-Down Hierarchy Generation** 节

### 2. 并行化方法

最早的方法是让每个线程处理相邻的两个莫顿码：
对第i个莫顿码和第i+1个莫顿码，若第一次在h位不同，则生成一个split，最终组成(i, h)，（i, h+1），...

最终会得到N - 1个莫顿码的split list。
此时对split list里的h排序，就拿到了莫顿码划分的顺序，也就拿到了BVH的空间划分。

该方法尽管已经一定程度上并行化，但最后建立BVH树时，即使拿到了序列，也必须按照序列顺序自顶向下逐层建树。

解决这一问题的方法在该文献中提出：https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

解决问题的思路从直接自顶向下建树转变为求每个内部节点的子节点范围

```cpp
Node* generateHierarchy( unsigned int* sortedMortonCodes,
                         int*          sortedObjectIDs,
                         int           numObjects)
{
    LeafNode* leafNodes = new LeafNode[numObjects];
    InternalNode* internalNodes = new InternalNode[numObjects - 1];

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.

    for (int idx = 0; idx < numObjects; idx++) // in parallel
        leafNodes[idx].objectID = sortedObjectIDs[idx];

    // Construct internal nodes.

    for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        int2 range = determineRange(sortedMortonCodes, numObjects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range.

        int split = findSplit(sortedMortonCodes, first, last);

        // Select childA.

        Node* childA;
        if (split == first)
            childA = &leafNodes[split];
        else
            childA = &internalNodes[split];

        // Select childB.

        Node* childB;
        if (split + 1 == last)
            childB = &leafNodes[split + 1];
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.

        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
    }

    // Node 0 is the root.

    return &internalNodes[0];
}
```
其中，`numObjects`是叶节点数，`sortedObjectIDs`为按莫顿码排序后的节点数组，`sortedMortonCodes`为排序后的莫顿码数组。

算法中最重要的部分是`determineRange`，该方法用于找到内部节点覆盖的子节点范围。要确定范围，只需要知道边界在何处。首先可以确定，任取一个节点，他一定属于某个包围盒的边界。因此接下来只需要解决两个问题：
1. 该节点所属包围盒应向左还是向右扩展
2. 扩展到何处停止

算法有方法 $\delta(i, j)$用于计算两个莫顿码的最长公共前缀长度。对于中间节点$I_i$，取莫顿码为$M_i$的叶节点为包围盒边界。则接下来比较$\delta(M_i, M_{i+1})$和$\delta(M_i, M_{i-1})$公共前缀长度更长的节点与$M_i$属于同一个包围盒。因为莫顿码的特性，公共前缀更长的节点一定距离更近。由此我们确定了中间节点$I_i$的上/下边界。

接下来考虑扩展应到何处停止。假设$M_i$向$M_{i+1}$方向扩展，我们记录$M_i$与$M_{i-1}$的最长公共前缀，然后计算其后所有节点与$M_i$的最长公共前缀，对任意节点$M_j$，若有$\delta(M_i, M_{i-1}) > \delta(M_i, M_j)$，则$M_{j-1}$即为边界。因为在当前包围盒内部，不应该存在一个节点，其距离比边界$M_i$ 到 $M_{i-1}$的距离还要长，该节点应该比$M_{i-1}$更早被分离出当前的包围盒

基于上述操作，我们就得到了每个内部节点的叶节点范围，基于此再寻找分割点，即可建立BVH


## 4. HLBVH
https://research.nvidia.com/sites/default/files/pubs/2010-06_HLBVH-Hierarchical-LBVH/HLBVH-final.pdf

Hierarchical LBVH(HLBVH)，本身的出发点是对早期的LBVH建树算法进行优化。文章认为LBVH本身对莫顿码的全局排序过于昂贵，因此将莫顿码排序分两个阶段进行。

### 1. 排序
算法伪代码见论文 **Figure 1**
其基本思想是：
1. 记莫顿码长度为3n，取其高位的3m作为上层粗粒度排序依据
2. 对高3m位，并行地看相邻序列是否一致，一致则分为一组
3. 按组对高位进行排序

e.g.
对莫顿码序列：
```
0           001001101010 | 010101...
1           001001101010 | 101010...
2           001001101010 | 111001...
3           001001101011 | 000111...
4           001001101011 | 010000...
5           001010000001 | 111111...
```
只看高位，分组有：
```
Run 0:
[ primitive 0, primitive 1, primitive 2 ]

Run 1:
[ primitive 3, primitive 4 ]

Run 2:
[ primitive 5 ]
```
然后选取run中的首个莫顿码为代表，对其进行排序即可

该过程被称为Compress-Sort-Decompress(CSD)

对于低位，则直接将高位排序后的数组分组放进并发线程数组（CTA），在shared memory里面并发排序。

该方法的优势在于，先只做上层排序，压缩了全局排序的任务规模，后续低位的排序
该方法的局限性在于其本身需要输入数据具有空间连续性

### 2. 建树
原始论文的方法对split的排序开销太大。HLBVH的思路是分步建树，每一步只看$p$位，从高位向低位建树，高位叶节点成为后续低位的根节点。这样的一个子树被称为一个深度为p的treelet

由于我们只看$p$位，因此每一层treelet最大只需要一个$2^p - 1$的数组就能存放其split关系

### SAH + HLBVH
HLBVH本身只是优化了建树的过程，实际的拓扑结构和LBVH建出来的是等价的，因此同样质量比较差。原因在于LBVH只看空间关系，不考虑AABB的重叠；此外，莫顿码是针对一个点的坐标生成的，对于较大的包围盒，只考虑单点做split，会导致实际的AABB跨多个split

优化方式非常直接，就是在顶层的粗粒度排序后，得到了BVH的所有下层子树后，对上层应用SAH，而不是简单的split。
同时，SAH在这里不再假设所有图元求交开销一致，而使用实际子树cluster的求交开销。

```
                N 个 primitives
                     │
              coarse Morton sort
                     │
                     ▼
               C 个 clusters
            ┌────────┼────────┐
            ▼        ▼        ▼
        cluster A cluster B cluster C
            │        │        │
          HLBVH    HLBVH    HLBVH
            │        │        │
            ▼        ▼        ▼
          subtree  subtree  subtree
            │        │        │
            └────────┼────────┘
                     │
             计算每个 subtree 的
                actual SAH cost
                     │
                     ▼
          把这些 subtree/cluster
          当成上层 SAH 的输入
                     │
                     ▼
                SAH top level
                     │
                     ▼
                 完整 BVH
```

## 5. TRBVH/ATRBVH
TRBVH 先用 LBVH 快速建树，再以每个内部节点为根选取局部 treelet，并用 SAH 改善其拓扑。

treelet 的边界节点称为 `treelet leaves`（frontier nodes）。从根开始，每次展开 frontier 中 AABB 表面积最大的可展开节点，直到达到目标大小：

```text
internal = {root}; frontier = {root.left, root.right}
重复：展开 frontier 中面积最大的内部节点
```

### 1. 原始 TRBVH

将每个 frontier node 看作不可拆分的子树，枚举所有二叉划分，选择局部 SAH 最小的拓扑。令 $B(S)$ 为包围子集 $S$ 的 AABB，则递推可简写为：

$$
D(S)=C_iA(B(S))+\min_{X\subset S}\left[D(X)+D(S\setminus X)\right]
$$

记录最优划分并回溯即可重建 treelet。该动态规划能求得局部最优，但复杂度随 treelet 大小指数增长，因此只能使用小 treelet（论文实现为 7 个 leaves、3 轮扫描）。

### 2. ATRBVH

ATRBVH 用贪心凝聚聚类替代 TRBVH 的穷举 DP：

1. 每个 treelet leaf 初始为一个 cluster；
2. 两个 cluster 的距离定义为合并后 AABB 的表面积：
   $$d(X,Y)=A\bigl(B(X)\cup B(Y)\bigr)$$
3. 反复合并距离最小的一对，并复用原 treelet 的内部节点连接它们；
4. 计算候选拓扑的 SAH，仅在新 SAH 更低时提交，否则丢弃。

ATRBVH 不保证 treelet 内全局最优，但复杂度更低，可以使用更大的 treelet。论文推荐 `treelet size = 9`、`iterations = 2`，平均构建时间比 TRBVH 少约 30%，而光追性能基本相当。

## 6. OIBVH
https://www.pure.ed.ac.uk/ws/files/142704991/Binary_Ostensibly_Implicit_CHITALU_DOA17022020_AFV.pdf
