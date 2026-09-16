# 蒙特卡洛积分

蒙特卡洛积分通过随机采样，把积分转化为期望的估计。它广泛用于无法解析求解的高维积分，例如路径追踪中的光传输积分。

## 1. 从积分到期望

设目标积分为

$$
I=\int_\Omega f(x)\,dx.
$$

在积分域 $\Omega$ 上选取概率密度 $p(x)$，并令随机变量 $X\sim p(x)$。要求在 $f(x)\neq 0$ 的区域内有 $p(x)>0$，则

$$
\begin{aligned}
I
&=\int_\Omega \frac{f(x)}{p(x)}p(x)\,dx \\
&=\mathbb E_{X\sim p}\left[\frac{f(X)}{p(X)}\right].
\end{aligned}
$$

因此，从 $p(x)$ 中独立同分布地抽取 $N$ 个样本 $X_1,\ldots,X_N$，可以构造估计量

$$
\boxed{
\hat I_N=\frac1N\sum_{i=1}^N\frac{f(X_i)}{p(X_i)}
}.
$$

其中 $p(x)$ 常称为采样分布或提案分布，$f(X_i)/p(X_i)$ 是样本的加权贡献。

## 2. 无偏性

在上述支持域条件成立且相关期望存在时，

$$
\begin{aligned}
\mathbb E[\hat I_N]
&=\frac1N\sum_{i=1}^N
\mathbb E\left[\frac{f(X_i)}{p(X_i)}\right] \\
&=\frac1N\sum_{i=1}^N I \\
&=I.
\end{aligned}
$$

所以该估计量是无偏的：重复进行估计时，其平均结果等于真实积分值。无偏并不代表单次估计的误差一定很小，噪声大小还取决于方差。

## 3. 大数定律与收敛

令

$$
Y_i=\frac{f(X_i)}{p(X_i)}.
$$

若 $Y_i$ 独立同分布且 $\mathbb E[|Y_i|]<\infty$，根据强大数定律，

$$
\hat I_N
=\frac1N\sum_{i=1}^N Y_i
\xrightarrow[N\to\infty]{a.s.}
\mathbb E[Y]
=I.
$$

也就是说，样本数趋于无穷时，估计结果几乎必然收敛到真实积分值。

## 4. 方差

先考察单个样本 $Y=f(X)/p(X)$。它的二阶矩为

$$
\begin{aligned}
\mathbb E[Y^2]
&=\int_\Omega
\left(\frac{f(x)}{p(x)}\right)^2p(x)\,dx \\
&=\int_\Omega\frac{f(x)^2}{p(x)}\,dx.
\end{aligned}
$$

由于 $\mathbb E[Y]=I$，单个样本的方差为

$$
\sigma^2
=\operatorname{Var}[Y]
=\int_\Omega\frac{f(x)^2}{p(x)}\,dx-I^2.
$$

对于 $N$ 个独立同分布的样本，

$$
\begin{aligned}
\operatorname{Var}[\hat I_N]
&=\operatorname{Var}\left[\frac1N\sum_{i=1}^N Y_i\right] \\
&=\frac1{N^2}\sum_{i=1}^N\operatorname{Var}[Y_i] \\
&=\frac{\sigma^2}{N}.
\end{aligned}
$$

代入 $\sigma^2$，得到

$$
\boxed{
\operatorname{Var}[\hat I_N]
=\frac1N\left(
\int_\Omega\frac{f(x)^2}{p(x)}\,dx-I^2
\right)
}.
$$

这里使用了独立样本之间协方差为零这一性质。一般情况下，

$$
\operatorname{Var}\left[\sum_iY_i\right]
=\sum_i\operatorname{Var}[Y_i]
+2\sum_{i<j}\operatorname{Cov}[Y_i,Y_j],
$$

而独立性使所有交叉协方差项消失。

## 5. 收敛速度、卡方散度与重要性采样

### 5.1 收敛速度

估计量的标准差为

$$
\operatorname{Std}[\hat I_N]=\frac{\sigma}{\sqrt N}.
$$

因此蒙特卡洛估计的典型误差以 $O(N^{-1/2})$ 的速度下降：若想把误差缩小一半，通常需要约四倍样本。不过，增加样本数并不是降低误差的唯一途径；选择更合适的采样分布 $p(x)$，也可以减小单个样本的方差 $\sigma^2$。

### 5.2 用卡方散度表示蒙特卡洛方差

为了量化采样分布 $p(x)$ 与理想分布之间的不匹配程度，可以把蒙特卡洛方差改写成卡方散度的形式。

仍考虑积分

$$
I=\int_\Omega f(x)\,dx,
$$

其中 $f(x)$ 可以取正值或负值。定义绝对值积分

$$
C=\int_\Omega |f(x)|\,dx,
$$

并假设 $0<C<\infty$。由此构造归一化概率密度

$$
q(x)=\frac{|f(x)|}{C}.
$$

显然 $q(x)\ge 0$ 且 $\int_\Omega q(x)\,dx=1$。同时要求提案分布 $p(x)$ 覆盖 $q(x)$ 的支持域，即 $q(x)>0$ 时必须有 $p(x)>0$。

由上一节得到蒙特卡洛估计量的方差

$$
\operatorname{Var}[\hat I_N]
=\frac1N\left(
\int_\Omega\frac{f(x)^2}{p(x)}\,dx-I^2
\right).
$$

因为 $f(x)^2=|f(x)|^2=C^2q(x)^2$，所以

$$
\int_\Omega\frac{f(x)^2}{p(x)}\,dx
=C^2\int_\Omega\frac{q(x)^2}{p(x)}\,dx.
$$

两个概率密度 $q$ 与 $p$ 之间的 Pearson 卡方散度定义为

$$
\chi^2(q\Vert p)
=\int_\Omega\frac{(q(x)-p(x))^2}{p(x)}\,dx.
$$

展开平方项，并利用 $\int_\Omega q(x)\,dx=\int_\Omega p(x)\,dx=1$，可得

$$
\begin{aligned}
\chi^2(q\Vert p)
&=\int_\Omega\frac{q(x)^2}{p(x)}\,dx
-2\int_\Omega q(x)\,dx
+\int_\Omega p(x)\,dx \\
&=\int_\Omega\frac{q(x)^2}{p(x)}\,dx-1.
\end{aligned}
$$

因此

$$
\int_\Omega\frac{q(x)^2}{p(x)}\,dx
=1+\chi^2(q\Vert p),
$$

代回方差公式，得到一般形式

$$
\boxed{
\operatorname{Var}[\hat I_N]
=\frac1N\left[
C^2\bigl(1+\chi^2(q\Vert p)\bigr)-I^2
\right]
}
$$

也可以写成

$$
\operatorname{Var}[\hat I_N]
=\frac{C^2}{N}\chi^2(q\Vert p)
+\frac{C^2-I^2}{N}.
$$

对于路径追踪中常见的非负被积函数 $f(x)\ge 0$，有 $C=I>0$，于是公式进一步化简为

$$
\boxed{
\operatorname{Var}[\hat I_N]
=\frac{I^2}{N}\chi^2(q\Vert p)
}.
$$

这说明：对于非负被积函数，蒙特卡洛方差与理想分布 $q$、提案分布 $p$ 之间的卡方散度成正比。若 $f$ 有正有负，则还会保留 $(C^2-I^2)/N$ 这一项，不能直接写成单纯的正比关系。

### 5.3 对重要性采样的解释

卡方散度满足 $\chi^2(q\Vert p)\ge 0$，并且仅当 $p=q$（几乎处处）时取零。因此理论上的最优提案分布为

$$
p^*(x)=q(x)=\frac{|f(x)|}{C}.
$$

当 $f(x)\ge 0$ 时，$p^*(x)=f(x)/I$，每个样本的贡献都相同：

$$
\frac{f(X)}{p^*(X)}=I,
$$

所以方差为零。实际问题中通常无法直接从 $p^*$ 采样，但这个结论给出了重要性采样的目标：让 $p(x)$ 的形状尽可能接近 $|f(x)|$，尤其避免在 $|f(x)|$ 较大的区域赋予过小概率。

如果 $p(x)$ 在重要区域过小，$f(x)/p(x)$ 会产生极端权重，卡方散度可能很大；若 $p$ 没有覆盖 $q$ 的支持域，则可以把 $\chi^2(q\Vert p)$ 视为无穷大，对应估计失效。

## 6. 在路径追踪中的对应关系

渲染方程中的半球积分可以写成

$$
L_r(x,\omega_o)
=\int_{\Omega^+}
f_r(x,\omega_i,\omega_o)
L_i(x,\omega_i)
|n\cdot\omega_i|
\,d\omega_i.
$$

若从方向分布 $p(\omega_i)$ 中采样入射方向，则单样本估计为

$$
\hat L_r
=\frac{
f_r(x,\omega_i,\omega_o)
L_i(x,\omega_i)
|n\cdot\omega_i|
}{p(\omega_i)}.
$$

路径追踪会递归地估计其中的入射辐亮度 $L_i$；BRDF 采样、光源采样和多重重要性采样，本质上都在尝试用更合适的采样分布降低这个估计的方差。

### 7.拓展到多重积分

假设现在不是一个随机变量，而是一组：
$$
X=(X_1,X_2,...,X_k)
$$
积分：
$$
I = \int ...\int f(x_1,...,x_k)dx_1...dx_k.
$$
定义联合PDF：
$$
p(x_1,...,x_k).
$$
同样乘除联合PDF：
$$
I = \int...\int \frac{f(x_1,...,x_k)}{p(x_1,...,x_k)}p(x_1,...x_k)dx_1...dx_k.
$$
所以：
$$
\boxed{I = E_{X\sim p}[\frac{f(X)}{p(X)}]}
$$
于是：
$$
\hat I = \frac{f(X)}{p(X)}
$$
这就是一个单样本多维的蒙特卡洛估计器
