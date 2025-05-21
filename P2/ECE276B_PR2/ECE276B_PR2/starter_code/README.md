# ECE276B PR2 Spring 2025

## Overview
In this assignment, you will implement and compare the performance of search-based and sampling-based motion planning algorithms on several 3-D environments.

### 1. main.py
This file contains examples of how to load and display the environments and how to call a motion planner and plot the planned path. Feel free to modify this file to fit your specific goals for the project. In particular, you should certainly replace Line 104 with a call to a function which checks whether the planned path intersects the boundary or any of the blocks in the environment.

### 2. Planner.py
This file contains an implementation of a baseline planner. The baseline planner gets stuck in complex environments and is not very careful with collision checking. Modify this file in any way necessary for your own implementation.

### 3. astar.py
This file contains a class defining a node for the A* algorithm as well as an incomplete implementation of A*. Feel free to continue implementing the A* algorithm here or start over with your own approach.

### 4. maps
This folder contains 7 test environments described via a rectangular outer boundary and a list of rectangular obstacles. The start and goal points for each environment are specified in main.py.


## Usage

### main.py

To use either **A Star** or **RRT Star**, please modify the main.py (if __name__ == "__main__" part), you can also sepcify different param for each planner. **You would need to install OMPL to run RRT Star**.


### utils.py

It contains implementation of collision checking and boundary checking for the part 1 of this project

### astar.py

It contains the implementation of the A star algorithm. The results and path images are in /astar folder

### rrt.py

It contains the RRT star algorithm which applies RRTstar from OMPL. The results and path images are in /rrt folder


太好了，你问得非常到位。我们就以 **x 轴这一维** 来说明 **Slabs 方法为啥有效**，你会发现它其实非常直观！

---

## ✅ Slabs 方法的基本思路：

我们想判断一条**线段**是否穿过一个 **AABB**，而这个 AABB 是「轴对齐」的立方体，意味着你可以分别在 x、y、z 三个轴上分别检查。

所以，我们在每个轴上都要回答一个问题：

> **这条线段有没有在这一维度上进入并穿出 AABB？**

---

## ✅ 现在只考虑 x 轴：

### 给定：

* 线段的起点和终点在 x 轴的投影是：

  $$
  x_0 = \text{p0}[0], \quad x_1 = \text{p1}[0]
  $$
* AABB 的 x 范围是：

  $$
  x_{\text{min}} = \text{block}[0], \quad x_{\text{max}} = \text{block}[3]
  $$

---

### 🎯 线段的参数化表达式（在 x 维）：

$$
x(t) = x_0 + t(x_1 - x_0), \quad t \in [0, 1]
$$

这表达了线段上任意点在 x 轴上的坐标。我们要找出这个表达式落在 AABB 的 $[x_{\min}, x_{\max}]$ 区间的 t 值。

---

### ✅ 求交点对应的 t 值（关键步骤）

#### step 1: 求线段穿过 $x_{\min}$ 和 $x_{\max}$ 的时间：

$$
t_1 = \frac{x_{\min} - x_0}{x_1 - x_0}, \quad t_2 = \frac{x_{\max} - x_0}{x_1 - x_0}
$$

这两个 t 值表示线段穿过 AABB 的 **进入点** 和 **离开点**（在 x 维度上）。

---

### ✅ step 2: 规范化

* 如果 $x_1 < x_0$，t1 会比 t2 大，所以我们取：

$$
t_{\text{enter}} = \min(t_1, t_2), \quad t_{\text{exit}} = \max(t_1, t_2)
$$

---

### ✅ step 3: 与其他维度取交集

我们要在 **x、y、z** 三个方向都“进入”和“退出”AABB，那线段穿过整个立方体的 t 范围是：

$$
t_{\text{min}} = \max(t_x^{\text{enter}}, t_y^{\text{enter}}, t_z^{\text{enter}})
$$

$$
t_{\text{max}} = \min(t_x^{\text{exit}}, t_y^{\text{exit}}, t_z^{\text{exit}})
$$

只有当 $t_{\text{min}} \leq t_{\text{max}}$ 时，线段才能在所有维度上同时“穿过” AABB。否则，说明它某一维根本没进到 AABB，就不可能整体穿过。

---

## ✅ 例子

### 例1：穿过

* 起点 $x_0 = 0$，终点 $x_1 = 10$
* AABB: $x \in [3, 7]$

$$
t_1 = (3 - 0)/10 = 0.3,\quad t_2 = (7 - 0)/10 = 0.7
$$

$$
t_{\text{enter}} = 0.3,\quad t_{\text{exit}} = 0.7
\Rightarrow t_{\text{min}} < t_{\text{max}} \Rightarrow \text{相交}
$$

---

### 例2：完全在外面

* 起点 $x_0 = 0$，终点 $x_1 = 2$
* AABB: $x \in [3, 7]$

$$
x(t) \in [0, 2] \Rightarrow 永远不在 [3, 7] 中，t_{\text{min}} > t_{\text{max}} \Rightarrow \text{不相交}
$$

---

## ✅ 为什么这个方法很高效？

* 它只用了 **基本的代数运算**
* 完全不需要生成线段上的点来“扫过去”
* 逻辑清晰，可扩展到 3D（就是多加几维）

---

如果你愿意，我可以画一张图示来帮助你可视化 x 轴上这两个交点是怎么计算出来的。要看吗？

