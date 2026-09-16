## 渲染方程的路径积分形式

首先我们忽略体积介质，只考虑表面渲染方程：
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
f_r(x_0,\omega_1,\omega_o),
\\
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
因为$L_i$实际上是下一个交点的$L_o$，所以沿着采样方向$w_1$从$x_0$发射ray，假设打到$x_1$，因为radiance沿真空传播不变，所以:
$$
L_i(x_0,w_1)=L_o(x_1,-w_1)
$$
于是：
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
其中：
$$
x_1=T(x_0,w_1)
$$
$x_1$是由$x_0$和$w_1$共同决定的，因此我在$x_1$交点构建的基于其半球法线的提案分布，本质上是一个条件分布$p(w_i^1|w_i^0,x_0)$。

随后我们把$x_1$处的渲染方程继续展开：

对于$x_1$:
$$
L_o(x_1,-\omega_1)
=
L_{e,1}
+
\int_{\Omega_1}
f_1c_1
L_o(x_2,-\omega_2)
\,d\omega_2.
$$
代回去得：
$$
L_o(x_0,\omega_o)
=
L_{e,0}
+
\int_{\Omega_0}
f_0c_0
\left[
L_{e,1}
+
\int_{\Omega_1}
f_1c_1
L_o(x_2,-\omega_2)
d\omega_2
\right]
d\omega_1.
$$
展开之后可以得到：
$$
\begin{aligned}
L_o
=&\;
L_{e,0}
\\
&+
\int_{\Omega_0}
f_0c_0L_{e,1}
\,d\omega_1
\\
&+
\int_{\Omega_0}
\int_{\Omega_1}
f_0c_0
f_1c_1
L_o(x_2,-\omega_2)
\,d\omega_2d\omega_1.
\end{aligned}
$$
这样便得到了第一次bounce后的路径积分，我们继续递归展开：
$$
\begin{aligned}
L_o
=&\;
L_{e,0}
\\
&+
\int
f_0c_0L_{e,1}
\,d\omega_1
\\
&+
\iint
f_0c_0f_1c_1L_{e,2}
\,d\omega_1d\omega_2
\\
&+
\iiint
f_0c_0f_1c_1f_2c_2L_{e,3}
\,d\omega_1d\omega_2d\omega_3
\\
&+\cdots
\end{aligned}
$$
这就是路径追踪的多重路径积分展开。

而路径追踪的能量来源便是每次bounce中的$L_{e,n}$,$L_{e,n}$可以是光源，也可以是天空盒。

#### 因此最终的路径追踪的路径积分形式可以写为：

一般第$k$项可以写成：
$$
\boxed{
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
}
$$
因此：
$$
L_o=\sum_{k=0}^{\infty}I_k
$$
这就是一种比较直观的path intergral展开形式。

#### 路径的蒙特卡洛估计器：

多维积分的蒙特卡洛方法及其推导见：[蒙特卡洛积分的数学推导](../Mathematic/monte-carlo.md)。

我们首先需要把“路径”当做一个随机变量，在路径追踪中，一条长度为k的路径可以表示成：
$$
\bar x = (x_0,x_1,...,x_k).
$$
或者如果我们用方向来参数化：
$$
\bar \omega = (w_1,w_2,...,w_k).
$$
于是我们可以认为：**一整条路径就是一个高维随机变量**

例如一个两bounce路径：
$$
C
\rightarrow x_0
\rightarrow x_1
\rightarrow x_2
$$
由$(w_1,w_2)$共同决定。

对于长度k的积分：
$$
I_k
=
\int\cdots\int
L_{e,k}
\prod_{j=0}^{k-1}f_jc_j
\,
d\omega_1\cdots d\omega_k.
$$
其蒙特卡洛估计器为：
$$
\hat I_k =\frac{L_{e,k}
\prod_{j=0}^{k-1}f_jc_j}{p(w_1,...,w_k)}
$$
而我们在相交表面基于局部法线方向构建的提案分布，从本质上来说是一个基于前几次弹射的条件分布，因此我们可以把联合分布写成：
$$
\boxed{
p(\omega_1,\omega_2,\ldots,\omega_k)
=
p(\omega_1)\,
p(\omega_2\mid\omega_1)\,
p(\omega_3\mid\omega_1,\omega_2)
\cdots
p(\omega_k\mid\omega_1,\ldots,\omega_{k-1})
}
$$
而实际上来说我们每次bounce中基于局部法线方向构建的提案分布，在路径视角来看都是由前N次bounce共同决定的**条件分布**

因此路径积分的蒙特卡洛估计器中的联合分布可以写成每次bounce中的局部视角的提案分布的乘积的形式：
$$
p(\omega_1,\ldots,\omega_k)
=
\prod_jp_j(\omega_{j+1})
$$
因此，第k长度路径的积分的蒙特卡洛估计器可以重新写为：
$$
\hat I_k =\frac{L_{e,k}
\prod_{j=0}^{k-1}f_jc_j}{\prod_{j=0}^{k-1}p_j(\omega_{j+1})}
$$
我们重新化简一下这个式子，为了简洁记号，我们首先令：
$$
p_j=p_j(\omega_{j+1}).
$$
上面的k长度路径的积分的蒙特卡洛估计器可以重新写为：
$$
\hat I_k
=
L_{e,k}
\prod_{j=0}^{k-1}
\frac{f_jc_j}{p_j}
$$
于是我们定义throughput:
$$
\beta_k = \prod_{j=0}^{k-1}\frac{f_jc_j}{p_j}.
$$
于是：
$$
\hat I_k = \beta_k L_{e,k}
$$
那么累积弹射$k$次的路径追踪的路径积分形式就可以写为：
$$
L_o=\sum_{k}\beta_kL_{e,k}.
$$
所以实际Path Tracing最核心的两个递推式其实就是：
$$
L\leftarrow L+\beta_k L_{e,k}
$$
和
$$
\beta_{k+1}=\beta_k \frac{f_kc_k}{p_k}.
$$
这也就是实际迭代式Path Tracer的数学形式，其伪代码可以表示为：

```c++
L = 0;
throughput = 1;

for each bounce k
{
    L += throughput * Le_k;

    throughput *= f_k * cosTheta_k / pdf_k;

    // sample next direction and trace to x_{k+1}
}
```

所以每次bounce可以非常简洁的理解为做了两件事：累加$\beta_kL_{e,k}$，更新$\beta_{k+1}$。

其中throughput $\beta_k$本质上就是**从相机到当前顶点之前所有Monte Carlo权重的乘积**