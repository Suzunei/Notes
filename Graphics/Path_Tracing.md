# Path Tracing篇

## 1.光线追踪

### 情况1：从相机原点朝屏幕像素发射光线

**此时的作用：从2D屏幕空间重建3D世界空间信息的查询工具**

光线追踪的理论基础其实是光路可逆，而屏幕中的每个像素其实可以视作是所有能够传入的光线的总和。我们难以准确地在世界空间中找寻所有射入成像平面的光，但是由于光路可逆原理，我们可以从向着成像平面的每个像素投射“视线”来向世界空间中投射光线，再计算光线命中位置的着色，并且模拟光线的多次弹射行为，就可以逆向找到那些向着屏幕发射了光线的世界空间中的点。

### 情况2：从世界空间中的任意一点朝着某些特定方向发射光线

**此时的作用：可见性与几何的查询工具**

从 Primary Ray 的角度来看，我们可以直观地把光线追踪理解成一种“从二维屏幕查询三维世界的工具”。但是这并非光线追踪更一般的定义，因为反射光线，阴影光线等 Secondary Ray 并不来源于屏幕。更一般地说，光线追踪是一种针对三维场景可见性与几何查询工具：给定空间中的一个起点和方向，查询该方向上与场景的相交关系。

**因此我们可以认为：**

最终：
$$
[ \boxed{ \text{Path Tracing} = 用 Ray Tracing 构造光路 + 用 Monte Carlo 估计光传输积分 } ]
$$
因为本章笔记为Path Tracing篇，因此对光追+光栅化的Hybrid pipeline不过多赘述，这个会放在GI篇中进行深入探讨。



## 2.路径追踪与蒙特卡洛方法

单一的光线追踪并不能实现着色，它最多只能算是一个屏幕空间的查询工具，它告诉我们：我们朝着这个像素看过去，能够看到什么，也就是我们只知道这个光线打在了什么物体上。因此它比起说是一种着色方法，我认为它更像是一种从 **2D 屏幕空间重建世界空间信息的查询工具**。

那我们在知道了视线将会打在什么物体之上时之后我们该怎么实现真正物理正确的着色呢？此时我们需要用到路径追踪。



### **路径追踪:**

路径追踪的基本思想简单来说就是，我们认为环境中的每一个点，都是一个独立的光源，他们会相互影响地照亮彼此，给环境中提供了非常多次弹射的间接光照——这个思路也是 GI 的核心思路之一。

而如何计算每一个点的亮度，我们需要引入辐照度理论来准确表示，此时我们需要搬出为一切现代的渲染奠定了基础的渲染方程：
$$
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o)\, L_i(x, \omega_i)\, (\omega_i \cdot n) \, d\omega_i
$$
对于渲染方程中的每一个变量：

- $x$: 表面交点，在Path Tracing中表示一条ray和场景求交得到的hit point，后续的法线N，材质，BRDF,入射光，出射光全是围绕和这个点$x$定义的。
- $n$: 表面法线（单位向量）
- $w_o$: 出射方向，其中的o表示outgoing，表示光从表面点$x$离开的方向
- $w_i$: 入射方向，其中的i表示incoming，表示一束入射光对应的方向
- $L_i(x,\omega_i)$: Incoming Radiance , 从方向$\omega_i$到达点$x$的Radiance
- Irradiance：把整个半球所有方向的$L_i(x,\omega_i)$乘上余弦项后积分${ E(x) = \int_{\Omega} L_i(x,\omega_i) (n\cdot\omega_i) \,d\omega_i }$
- $L_e(x,\omega_o)$: Emitted Radiance ,自发光亮度，点$x$自己沿$\omega_o$方向发射的Radiance
- $f_r(x,\omega_i,\omega_o)$: BRDF,双向反射分布函数，它回答的问题是：从$\omega_i$方向进入的光，有多少会被表面反射到$\omega_o$
- ($\omega_i\cdot n$):余弦项（$\cosine\theta_i$）描述的入射光与法线的夹角，同时也是入射光的投影面积效应，比如一个光正对着照，其贡献最大，如果几乎擦着表面照，其贡献趋近于0，同时$\cosine\theta$也趋近于0
- $\Omega$:积分区域，这里一般指表面点$x$上方的单位半球，也就是所有可能的入射方向：$\omega_i\in\Omega$,所以Rendering Equation做的事情实际上就是：把上半球中所有方向过来的光全部累加起来
- $d\omega_i$:微分立体角

因此整个积分代表的是：把上半球面所有可能的入射方向对出射方向的贡献全部累加起来

然后用一句话来概括Rendering Equation就是：

> 点$x$沿$\omega_o$方向离开的光，等于这个点自己发出的光，加上来自上半球所有方向的入射光，经过余弦和材质BRDF调制后反射到$\omega_o$方向的总和

这样配合RayTracing的确就可以在弹射次数足够多的情况下准确算出每个点的Radiance和Irradiance，但是这个渲染方程本身是一个高维的积分，并且是一个递归的函数，光线还可能会分裂...，我们几乎不可能实时地去求解它。

因此我们需要一个方法高效地来估算这个积分的值，于是我们使用了蒙特卡洛积分。

### 递归路径追踪：

递归路径追踪的核心是：

> $L_i$实际是下一个交点的$L_o$，于是渲染方程可以写成递归的形式

忽略体积介质，考虑表面的渲染方程：
$$
L_o(x_0,\omega_o)
=
L_e(x_0,\omega_o)
+
\int_{\Omega_0}
f_r(x_0,\omega_1,\omega_o)
L_i(x_0,\omega_1)
\cos\theta_1
\,d\omega_1.
$$
为了简化符号，定义：
$$
f_0
=
f_r(x_0,\omega_1,\omega_o),\\
c_0
=
|\cos\theta_1|
=
|n_0\cdot\omega_1|.
$$
于是：
$$
L_o(x_0,\omega_o)
=
L_{e,0}
+
\int_{\Omega_0}
f_0c_0
L_i(x_0,\omega_1)
\,d\omega_1.
$$
而$L_i$实际是下一个交点$x_1$的$L_o$，则有：
$$
L_i(x_0,\omega_1)
=
L_o(x_1,-\omega_1).
$$
于是我们把所有的$L_i$写成下一级的$L_o$的形式：
$$
L_o(x_0,\omega_o)
=
L_{e,0}
+
\int_{\Omega_0}
f_0c_0
L_o(x_1,-\omega_1)
\,d\omega_1.
$$
于是此时我们得出了渲染方程的递归形式。

于是我们将渲染方程递归展开，展开过程见：[蒙特卡洛积分的数学推导](../Mathematic/monte-carlo.md)，最终得到：

展开后的第k项可以写为：
$$
I_k
=
\int
\cdots
\int
L_{e,k}
\prod_{j=0}^{k-1}
f_jc_j
\,
d\omega_1\cdots d\omega_k
$$
因此最终渲染方程的路径积分展开形式为：
$$
L_o
=
\sum_{k=0}^{\infty}I_k
$$

### 蒙特卡洛方法：

蒙特卡洛积分把积分改写为期望，并通过随机采样的平均值进行估计。若 $X_i\sim p(x)$，则常用估计量为：

$$
\hat I_N=\frac1N\sum_{i=1}^N\frac{f(X_i)}{p(X_i)}.
$$

在满足支持域和可积性等条件时，该估计量无偏并会随样本数增加而收敛；其标准差按 $O(N^{-1/2})$ 下降。

对于路径追踪中常见的非负被积函数，令

$$
I=\int_\Omega f(x)\,dx,
\qquad
q(x)=\frac{f(x)}{I},
$$

其中 $q(x)$ 是由被积函数归一化得到的理想目标分布。蒙特卡洛方差可以写成 Pearson 卡方散度的形式：

$$
\boxed{
\operatorname{Var}[\hat I_N]
=\frac{I^2}{N}\chi^2(q\Vert p)
}
$$

其中

$$
\chi^2(q\Vert p)
=\int_\Omega\frac{(q(x)-p(x))^2}{p(x)}\,dx.
$$

当积分 $I$ 和样本数 $N$ 固定时，降低蒙特卡洛方差就等价于减小提案分布 $p$ 与理想目标分布 $q$ 之间的卡方散度。因为 $\chi^2(q\Vert p)\ge 0$，并且仅在 $p=q$（几乎处处）时取零，所以理想的提案分布为

$$
p^*(x)=q(x)\propto f(x).
$$

因此，**重要性采样的本质就是选择一个更接近被积函数形状的提案分布，从而减小卡方散度和估计方差**。实际中通常无法直接使用最优分布，只能利用 BRDF、光源分布等已知信息构造对它的近似。

完整证明、方差推导以及它与路径追踪的对应关系见：[蒙特卡洛积分的数学推导](../Mathematic/monte-carlo.md)。



### 路径追踪中的蒙特卡洛方法：

如果要使用蒙特卡洛方法来求解渲染方程，那我们首先需要渲染方程中的哪些部分是已经确定的，哪些部分是随机变量：

我们回到渲染方程本身：
$$
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o)\, L_i(x, \omega_i)\, (\omega_i \cdot n) \, d\omega_i
$$
于是我们可以分析渲染方程在蒙特卡洛路径追踪中的特点：

（1）蒙特卡洛方法只用于估计积分的部分，因此$L_e$不纳入我们的考虑之中，因此我们只需要考虑积分中出现的变量$x$,$w_i$,$w_o$,$n$。

（2）而在路径追踪中，注意因为我们是从相机发射光线去反向追踪入射光，所以我们发射的光线对应的是$w_o$,而bounce出去的光线才是$w_i$。

（3）而每次光线与物体表面求交时，入射光线$w_o$，交点位置$x$，物体表面属性$(n,f_r)$都已经确定。

因此：

> 在其他条件已经确定的前提下，渲染方程这个局部积分的积分变量只有$w_i$。在蒙特卡洛估计中，我们将$w_i$视为从某个PDF $p(w_i)$中采样得到的随机变量。

因此蒙特卡洛路径追踪的核心，就是按照某一概率密度$p(w_i)$对$w_i$进行随机采样，并通过这些方向样本估计半球积分。

因此：

> 局部来看随机变量是$w_i$: 整条路径来看随机变量是一系列$w_1$,$w_2$,...

既然当前Bounce唯一需要选择的积分变量就是$w_i$,那么降低方差的核心问题，就变成了**如何设计$p(w_i)$,使它尽可能接近integrand的形状**
$$
p^*(\omega_i)
\propto
f_r(x,\omega_i,\omega_o)
L_i(x,\omega_i)
(\omega_i\cdot n).
$$
这也是后续Importance samping和RIS的出发点



## 3.重要性采样

考虑上述我们要解决的核心问题，我们希望
$$
p^*(w_i)\propto f_r(x,w_i,w_o)L_i(x,w_i)(w_i\cdot n)
$$
直接正比于后面那一大块比较困难，**但是我们可以先简单正比于其中的主导项，也就是BRDF**

### 对BRDF的重要性采样

对BRDF的重要性采样需要按照BRDF的类型来区分：

#### （1）Lambert漫反射的重要性采样

漫反射的BRDF为：
$$
f_r = \frac{\rho}{\pi}
$$
它是一个常数，和$w_o$,$w_i$都无关

因此我们需要估计的积分是：
$$
I
=
\int_{\Omega}
\frac{\rho}{\pi}
L_i(\omega_i)
\cos\theta_i
\,d\omega_i.
$$
令$g(\omega_i) = \frac{\rho}{\pi} L_i(\omega_i) \cos\theta$，我们想让$p^*(w_i)\propto g(w_i)$,我们可以直接忽略掉前面的BRDF项，因为是个常数，于是我们只需要正比于后面的$L_i(w_i)\cosine \theta_i$即可。

但是直接正比于这个也很困难，于是我们假设一种理想情况$L_i(w_i)$同样是个常数，那我们只需要构造一个正比于$\cosine\theta_i$的分布即可

于是我们构造一个$p(\omega_i)$，并希望$p(\omega_i)\propto cos\theta_i$，同时因为我们是在上半球面积分，$\theta_i$的范围为：$0<\theta_i<\frac{\pi}{2}$ ，$cos\theta_i$（入射光线和法线的点乘）一定是非负的。

于是得出我们需要去求的$p(\omega_i)$需要满足的两个条件：

（1）$p(\omega_i)$需要具备PDF都具备的归一化，绝对非负的特性

（2）$p(w)\propto cos\theta_i$

令$p(\omega)=\frac{cos\theta_i}{C}$, 此时我们只需要满足$\int_\Omega p(\omega)d\omega=1$即可，最后可以解得$C=\frac{1}{\pi}$，具体的证明过程在：[重要性采样的数学推导](../Mathematic/importance-sampling.md)。

因此可以得出Cosine分布的PDF为：
$$
p(\omega)=\frac{cos\theta}{\pi}.
$$
有了PDF之后我们可以进行逆变换采样，我们需要生成两个U[0,1]的样本，然后通过求逆变换的方式得到服从Cosine分布的样本

我们令：$\xi_1,\xi_2\sim U[0,1].$

最终通过求逆变换得到的样本$\omega$为：
$$
\omega=
\begin{pmatrix}
\sqrt{\xi_1}\cos(2\pi\xi_2)\\
\sqrt{\xi_1}\sin(2\pi\xi_2)\\
\sqrt{1-\xi_1}
\end{pmatrix}
$$
具体的求逆变换的推导过程在：[重要性采样的数学推导](../Mathematic/importance-sampling.md)。

因为$\cosine\theta=max(0,n\cdot \omega)$，所以原式在$n$确定的情况下可以写成：
$$
p(\omega)=\frac{max(0,n\cdot \omega)}{\pi}
$$
设我们采样的样本为$w_i \sim p(w)$,则此时我们的蒙特卡洛估计量为：
$$
\boxed{
\hat I(\omega_i)=\rho\,L_i(\omega_i)
}
$$
如果是N个独立样本，则：
$$
\boxed{
\hat I_N
=
\frac1N
\sum_{k=1}^N
\rho\,L_i(\omega_i^{(k)})
}
$$
如果算上自发光，则当前像素，当前bounce的估计可以写成：
$$
\boxed{
\hat L_o
=
L_e(x,\omega_o)
+
\frac1N
\sum_{k=1}^N
\rho\,L_i(\omega_i^{(k)})
}
$$
