# Jackaroo 实现说明

## 1. 训练目标

Jackaroo 把完整足球比赛分解为连续的进球 segment。第一个 segment 从比赛开球开始，此后每粒进球同时结束当前 segment 并开始下一次开球。己方率先取得下一球时，segment 结果为 $+1$；对方率先取得下一球时，结果为 $-1$。比赛到达时间上限时，最后一个无进球尾段进入比赛统计，并从优化数据中移除。

这个定义使每段轨迹都具有明确的胜负结果：

$$
G_k\in\{-1,+1\}.
$$

segment 内采用 $\gamma=1$，所以该段的每个有效决策共享同一个 Monte Carlo 回报 $G_k$。比分和绝对比赛时间由环境负责记录与终止比赛；策略输入聚焦当前可观测局面。

Jackaroo 采用单智能体控制。训练采用共享策略自对弈，同一个 Jackaroo 网络分别接收左右两队的规范化观察，并在每一步为双方各自的 `designated_possession_player` 选择一个原子动作。每队其余球员继续由 TeamAI 与 Eliza 驱动。引擎在收齐双方决策后统一执行完整足球规则、物理、动画和球队行为，并分别返回两侧的下一步公开观察。

## 2. 观察与动作

每个观察包含球的位置、速度和旋转，双方二十二名球员的位置、速度、朝向、阵型位置、动作阶段、疲劳、触球及到球时间，球队球权状态，比赛模式，定位球状态，当前球权与最近触球信息，以及持续动作状态。训练编码移除累计比分、绝对时间和引擎步数，使策略价值始终表示当前局面下的下一球期望。

双方球员、球和比赛上下文构成 24 个实体 token。双方物理侧在观察进入策略前被统一为己方向正 $x$ 方向进攻，因此同一策略可以交替控制左右两侧。

策略动作直接对应引擎的 32 个原子动作：

$$
a_t\in\{0,\ldots,31\}.
$$

动作表示由二维方向、足球功能类别和按下/释放阶段组成。八个移动方向共享方向结构，同一足球按钮的按下与释放共享功能身份。该表示让相近动作在训练中共享统计规律。

## 3. 持久实体策略网络

实体 Transformer 首先在每个决策步建模二十二名球员、球和比赛上下文之间的关系。三个 8 头自注意力 block 产生 128 维实体表示，并汇总当前受控球员、双方球队、球和比赛上下文。

随后，每个实体使用共享参数的 GRUCell 更新自己的 64 维持久状态，完整局面使用另一个 GRUCell 更新 256 维全局状态：

$$
h_t^i=\mathrm{GRU}_{\mathrm{entity}}(e_t^i,h_{t-1}^i),
$$

$$
h_t^{\mathrm{global}}
=
\mathrm{GRU}_{\mathrm{global}}(g_t,h_{t-1}^{\mathrm{global}}).
$$

实体状态始终附着于同一名球员，因此 `designated` 从球员 A 变为球员 B 时，双方的跑动趋势、动作进度和历史关系仍保持各自身份。每个进球 segment 结束时，训练器清空该环境的全部循环状态。

PPO 使用长度为 32 的连续序列执行 truncated backpropagation through time。每个序列保存真实的起始循环状态和 padding mask，序列边界始终位于同一个进球 segment 内。

## 4. 空间控制与动作结果

### 4.1 DSS 控制先验

空间控制分支在 $12\times8$ 个球场位置上估计双方球员的可达性。对于球员 $i$ 和位置 $x_k$，模型根据当前位置 $p_i$、速度 $v_i$、运动方向和疲劳计算 DSS 风格的基础到达时间：

$$
r_{ik}=\lVert x_k-p_i\rVert,
\qquad
\cos\theta_{ik}
=
\frac{v_i^{\mathsf T}(x_k-p_i)}
{\lVert v_i\rVert r_{ik}+\varepsilon},
$$

$$
T^{\mathrm{DSS}}_{ik}
=
\frac{r_{ik}}
{\max\left(\varepsilon,
(\lVert v_i\rVert\cos\theta_{ik}+V_i)/2\right)}.
$$

GameplayFootball 的离散动作和动画阶段会改变实际到达过程，因此实体 token 为基础时间提供正值修正：

$$
T^{\mathrm{eng}}_{ik}
=
T^{\mathrm{DSS}}_{ik}
\exp r_\eta(h_t^i,x_k-p_i).
$$

双方球队的 soft minimum 到达时间随后生成位置控制程度：

$$
C_t(x_k)
=
\sigma\left(
\frac{\widetilde T_{\mathrm{opp},k}-\widetilde T_{\mathrm{own},k}}
{\tau_C}
\right).
$$

训练器使用公开观察中的 `time_to_ball_ms` 校准到球位置的预测时间。这个监督把 DSS 运动先验适配到引擎实际轨迹，同时让 96 个位置 token 表达双方当前能够控制的区域及其贡献球员。

### 4.2 动作条件转移

动作结果模型同时评价 32 个候选动作。它为每个动作预测下一步全局隐状态、216 维公开结果向量和三类边界概率。公开结果向量由球的运动、二十二名球员的位置与速度、空间控制场、球权和下一决策节点组成。每项预测都以当前观察历史 $h_t$ 与候选动作 $a_t$ 为条件，因此模型表达的是 $P(s_{t+1}\mid h_t,a_t)$ 所描述的局面相关因果关系。同一组动作在不同局面中会产生不同结果，进球链由连续的真实局面转移构成。

已执行动作与真实相邻观察提供监督。连续量使用均方误差，隐状态使用 smooth L1，边界类别使用交叉熵：

$$
\mathcal L_{\mathrm{transition}}
=
\lambda_z\lVert\widehat z_{t+1}-z_{t+1}\rVert_2^2
+
\lambda_h\mathrm{SmoothL1}(\widehat h_{t+1},h_{t+1})
+
\lambda_d\mathrm{CE}(\widehat d_{t+1},d_{t+1}).
$$

这组目标使用实际执行结果训练动作后果表示；所有 rollout 仍由真实引擎产生。

## 5. 单一 EPV 与动作价值

全局循环状态进入一个 EPV 头，输出当前策略下取得下一球的期望结果：

$$
V_\phi(s_t)
=
\mathbb E_\pi[G_k\mid s_t],
\qquad
V_\phi(s_t)\in[-1,1].
$$

这个 $V_\phi$ 同时承担 PPO Critic 和控制结果价值函数。它使用真实 segment 结果进行回归：

$$
\mathcal L_V
=
\left(V_\phi(s_t)-G_k\right)^2.
$$

动作结果模型为动作 $a$ 产生继续、己方进球和对方进球的概率 $p_c^a,p_+^a,p_-^a$，并预测下一隐状态 $\widehat h_{t+1}^a$。由同一个 EPV 推导的单步动作价值为：

$$
Q_\phi(s_t,a)
=
p_+^a-p_-^a+p_c^aV_\phi(\widehat h_{t+1}^a).
$$

因此，$Q_\phi$ 是由转移预测和唯一 EPV 组成的 Bellman 动作价值，而不是另一套价值网络。已执行动作的 $Q_\phi(s_t,a_t)$ 同样回归真实 $G_k$，从而校准动作结果概率与下一状态 EPV 的组合。

Actor 使用完整实体关系、结构化动作表示和预测公开结果生成基础 logit，并通过一个初始值为零的可学习门将 $Q_\phi$ 加入 logit。预测结果和动作价值在 Actor 路径中停止梯度，因此 PPO 负责学习策略偏好，真实相邻观察和 segment 结果负责训练转移与价值语义。

## 6. PPO 与数据流

采样器让 C++ 向量环境以 flight 形式并行推进完整比赛，并在每个决策步把所有活跃比赛的左右观察合并为一次 GPU 推理。正式配置的单个 flight 包含 16 场比赛；一次更新连续运行多个 flight，直到完成 256 场比赛。每场比赛最多推进 3000 个引擎步，双方在每个有效步骤各自产生一次策略决策。进球会同时结束两侧的当前 segment，并产生结果互为相反数的两条训练轨迹。比赛结束时尚未产生下一粒进球的尾段结束于 $0$，该尾段退出 PPO 数据集。一次更新保留的进球 segment 数由 256 场比赛中的实际进球数决定。

对有效 transition，优势为：

$$
\widehat A_t=G_k-V_\phi(s_t).
$$

PPO 使用旧策略记录的 log probability 计算概率比，并采用标准 clipped objective：

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

$$
\mathcal J_{\mathrm{clip}}
=
\mathbb E_t\left[
\min\left(
\rho_t\widehat A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
$$

最终训练目标把策略、单一 EPV、动作结果、动作价值和控制场校准合并为：

$$
\mathcal L
=
-\mathcal J_{\mathrm{clip}}
+c_V\mathcal L_V
-c_H\mathcal H(\pi_\theta)
+c_T\mathcal L_{\mathrm{transition}}
+c_Q\mathcal L_Q
+c_C\mathcal L_{\mathrm{control}}.
$$

近似 KL 与 clipping fraction 只作为更新幅度的诊断指标。每次更新还记录策略损失、EPV 损失、策略熵、转移损失、动作价值损失、控制场损失、动作分布、进球 segment 数、删失尾段数量和完整自对弈比分。

## 7. 训练与评估

正式训练从随机初始化参数开始。训练 seed 初始化一个可复现的伪随机序列，每场 self-play 比赛从该序列取得唯一的 32 位引擎 seed。一个共享 Jackaroo 网络同时扮演两侧，使每粒进球都提供正负成对的策略样本，并让双方动作都成为可学习决策。每次 PPO 更新完成后，策略使用固定 seed 42 与内置 AI 进行两场完整验证，Jackaroo 分别位于左右物理侧。验证采用贪心动作、记录整场比分，并把两场比赛写入同一个 GFR 文件；验证轨迹不进入训练数据，也不按 segment 计分。

检查点保存网络、优化器、网络结构标识、观察与动作维度、序列长度、训练 seed、损失权重、更新编号和最新诊断。恢复训练时，加载器校验这些决定张量语义的字段，然后继续更新。

训练结束后，导出器把策略编译为 TorchScript。部署接口接收一个公开观察、24 个实体循环状态和全局循环状态，并返回 32 个动作 logit、EPV 及更新后的循环状态。

## 8. 未来拓展

固定训练 seed 上稳定获胜后，可以逐步加入多个训练 seeds，并继续使用固定评估集合区分记忆与泛化。策略具备可靠原子动作能力后，还可以加入球员选择头，把动作扩展为球员索引与原子动作的联合决策，并继续使用进球 segment 的 PPO 回报训练决策节点。

模型层还可以利用动作条件转移表示开展短时规划实验。规划器可以比较候选动作序列的预测 EPV，再将选择结果用于在线决策或策略蒸馏。该方向沿用同一套公开观察、32 个原子动作、进球 segment 目标和唯一 EPV 语义。
