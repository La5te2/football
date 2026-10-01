# Jackaroo 实现说明

## 1. 训练目标

Jackaroo 把完整足球比赛分解为连续的进球 segment。第一个 segment 从比赛开球开始，此后每粒进球同时结束当前 segment 并开始下一次开球。己方率先取得下一球时，segment 结果为 $+1$。对方率先取得下一球时，结果为 $-1$。比赛到达时间上限时，最后一个结果为 $0$ 的尾段进入比赛统计，优化数据保留结果为 $+1$ 或 $-1$ 的 segment。

这个定义使每段轨迹都具有明确的胜负结果：

$$
G_k\in\{-1,+1\}.
$$

segment 内采用 $\gamma=1$，所以该段的每个有效决策共享同一个 Monte Carlo 回报 $G_k$。比分和绝对比赛时间由环境负责记录与终止比赛。策略输入聚焦当前可观测局面。

本文把“下一球价值”定义为：从当前局面继续采用现有策略时，己方先取得下一球与对方先取得下一球之间的期望结果。对于局面 $s_t$，其定义为：

$$
V(s_t)=\mathbb E[G_k\mid s_t],
\qquad V(s_t)\in[-1,1].
$$

$V(s_t)$ 越接近 $+1$，己方越可能率先取得下一球。$V(s_t)$ 越接近 $-1$，对方越可能率先取得下一球。$V(s_t)$ 接近 $0$ 表示双方机会接近。后文的价值头负责从网络内部状态估计这个量。

TeamAI 会根据比分、比赛时间和近期控球调整球队行为。Jackaroo 通过球员状态、动态阵型和实际运动读取这些调整的外显结果，因此其决策过程具有部分可观测性。持久实体状态和全局状态沿 segment 连续传播，用于从公开轨迹估计当前球队倾向及相关隐藏条件。

Jackaroo 采用单智能体控制。训练采用共享策略自对弈，同一个 Jackaroo 网络分别接收左右两队的规范化观察，并在每一步为双方各自的 `designated_possession_player` 选择一个原子动作。每队其余球员继续由 TeamAI 与 Eliza 驱动。引擎在收齐双方决策后统一执行完整足球规则、物理、动画和球队行为，并分别返回两侧的下一步公开观察。

## 2. 观察与动作

每个观察包含球的位置、速度和旋转，双方二十二名球员的位置、速度、朝向、阵型位置、动作阶段、疲劳、触球及到球时间，球队球权状态，比赛模式，定位球状态，当前球权与最近触球信息，以及持续动作状态。训练编码移除累计比分、绝对时间和引擎步数，使策略价值始终表示当前局面下的下一球期望。

双方球员、球和比赛上下文构成 24 个实体 token。这里的实体 token 是表示一个可观测对象的固定宽度向量：二十二个 token 分别表示二十二名球员，一个表示球，一个表示比赛模式与球权等全局上下文。双方物理侧在观察进入策略前被统一为己方向正 $x$ 方向进攻，因此同一策略可以交替控制左右两侧。

策略动作直接对应引擎的 32 个原子动作：

$$
a_t\in\{0,\ldots,31\}.
$$

引擎公开状态为当前队伍给出决策节点：

$$
d_t=D(s_t),\qquad d_t\in\{0,\ldots,10\}.
$$

“决策节点”就是该步骤由外部策略直接控制的球员索引。引擎根据当前球权竞争与球队状态给出这个索引，策略随后为对应球员选择动作。

训练环境把十一行动作数组初始化为 `game_delegate`，并把 $a_t$ 写入第 $d_t$ 行。于是 Jackaroo 每一步直接决定当前节点的动作，其余十名球员继续由 TeamAI 与 Eliza 驱动。

固定的引擎决策节点把训练样本集中在持球、接球、争球和关键防守等高影响局面。Jackaroo 在这些位置承担动作结果，并通过进球 segment 回报学习处理球和改变球队空间关系。

每个动作使用 18 维固定结构特征：二维方向向量 $u_a$、13 维功能类别 $c_a$ 和 3 维空闲/按下/释放阶段 $q_a$。其中，动作 0 为空闲，动作 1 至 8 是八个方向的按下，动作 9 至 19 是十一个足球功能的按下，动作 20 是方向释放，动作 21 至 31 是对应足球功能的释放。动作编码器把这组特征映射为 64 维 embedding：

$$
x_a=[u_a,c_a,q_a]\in\mathbb R^{18},
\qquad
e_a=f_{\mathrm{action}}(x_a)\in\mathbb R^{64}.
$$

这里的动作 embedding 是由可学习映射产生的固定宽度动作向量。八个移动方向通过 $u_a$ 表达方向邻近关系，足球功能通过 $c_a$ 区分，持续动作的按下与释放共享功能类别并由 $q_a$ 区分阶段。该表示让相邻方向、相同足球功能和成对的按下/释放动作共享统计规律。

## 3. 持久实体策略网络

Entity Transformer 首先在每个决策步建模二十二名球员、球和比赛上下文之间的关系。它采用三个 8 头 self-attention block。self-attention 根据当前状态学习每对实体之间的权重，使每个实体表示吸收与其决策相关的其他实体信息。Entity Transformer 最终产生 128 维实体表示，并汇总当前受控球员、双方球队、球和比赛上下文。

随后，每个实体使用共享参数的 GRUCell 更新自己的 64 维持久状态，完整局面使用另一个 GRUCell 更新 256 维全局状态。GRUCell 是带门控的循环单元，它把当前输入与上一步状态结合，并通过可学习的门选择进入新状态的历史信息。实体状态始终附着于同一名球员，因此 `designated` 从球员 A 变为球员 B 时，双方的跑动趋势、动作进度和历史关系仍保持各自身份。全局状态保存跨实体的局面历史。每个进球 segment 结束时，训练器清空该环境的全部循环状态。

二十二名球员、球和比赛上下文各自保存一个 64 维实体状态。二十四个实体共享同一个 Entity GRUCell 参数。状态为 inactive 的球员更新结果归零。Global GRUCell 的输入同时汇合扁平公开观察、Entity Transformer 关系摘要和空间控制摘要：

$$
\widetilde h_t^i
=
\mathrm{GRU}_{\mathrm{entity}}\left(
W_e e_t^i,h_{t-1}^i
\right),
\qquad
h_t^i=m_t^i\widetilde h_t^i,
$$

$$
u_t
=
f_{\mathrm{obs}}(s_t)
+f_{\mathrm{relation}}(E_t)
+f_{\mathrm{control}}(C_t),
$$

$$
h_t^{\mathrm{global}}
=
\mathrm{GRU}_{\mathrm{global}}\left(
u_t,
h_{t-1}^{\mathrm{global}}
\right),
$$

其中 $m_t^i$ 是实体有效性 mask。mask 是取值为 $0$ 或 $1$ 的指示量，状态为 inactive 的球员取得 $0$。$E_t$ 是 Entity Transformer 关系摘要，$C_t$ 是由 96 个球场位置 token 加权汇总得到的网格表示。位置 token 是表示一个固定网格位置的向量，其中包含该位置的控制关系、空间质量和最可能到达该处的球员状态。当前决策节点的实体 token、持久实体状态、全局状态和网格表示随后组成 Actor 的关系表示。第 4 节给出位置 token、权重和汇总过程的完整定义。

PPO 使用长度由 `--sequence-length` 指定的连续序列执行 truncated backpropagation through time，默认长度为 32。每个序列保存采样时真实的起始循环状态 $(H_{t_0-1},h_{t_0-1}^{\mathrm{global}})$ 和有效步骤 mask。序列在同一个进球 segment 内切分。padding 步骤的 mask 取 $0$，进球后两侧的实体状态和全局状态同时归零。因此，反向传播长度受到控制，而正向隐藏状态仍沿完整 segment 连续传递。

## 4. 空间控制与动作结果

“到达时间”表示一名球员从当前的位置、速度和朝向出发，在当前疲劳状态下到达指定球场位置所需的归一化模拟时间。较小的到达时间表示该球员能更早影响该位置。

“球队控制概率”表示双方竞争同一位置时，己方先到达并控制该位置的相对程度，记为 $C_t(x_k)\in[0,1]$。数值接近 $1$ 表示己方占优，接近 $0$ 表示对手占优，接近 $0.5$ 表示双方到达能力接近。

“位置重要性”表示控制一个位置对当前进攻方向和球状态的相关程度，记为 $W_t^q(x_k)\in[0,1]$，其中 $q$ 表示己方或对手。它随球的位置、球的速度和球队进攻方向变化。

“空间质量”表示一支球队对重要位置实际拥有的控制量。己方空间质量等于己方位置重要性乘以己方控制概率。对手空间质量等于对手位置重要性乘以对手控制概率。这个乘积使网络把参数集中在同时具备局面意义和实际可控性的区域。

空间分支把受控球员动作、其他球员响应、控制关系变化、空间质量变化和下一球价值组织为同一条可训练因果链：

$$
\text{受控球员动作}
\rightarrow
\text{其他球员响应}
\rightarrow
\text{空间控制关系变化}
\rightarrow
\text{重要区域的空间质量变化}
\rightarrow
\text{下一粒进球的概率变化}.
$$

Jackaroo 首先估计每名球员对每个网格位置的到达时间，再把双方到达时间转换成连续的球队控制概率。状态相关的位置重要性与控制概率相乘后形成空间质量，包含空间质量的完整状态最终进入唯一的下一球价值头。公开引擎轨迹负责校准到达时间和位置重要性，真实进球 segment 负责监督第 1 节定义的下一球价值。这些步骤共同组成一条内部数据流。

### 4.1 方向相关的到达时间与球队控制

空间控制分支在 $12\times8$ 个球场位置上估计双方球员的可达性。对于球员 $i$ 和位置 $x_k$，模型根据当前位置 $p_i$、速度 $v_i$、运动方向和疲劳计算基础到达时间。实现使用 $V_i=0.012(1-0.35f_i)$ 表示疲劳 $f_i$ 调整后的最大速度，并使用归一化系数 $\kappa=10^{-3}$：

$$
r_{ik}=\lVert x_k-p_i\rVert,
\qquad
\cos\theta_{ik}
=
\frac{v_i^{\mathsf T}(x_k-p_i)}
{\lVert v_i\rVert r_{ik}+\varepsilon},
$$

$$
A_{ik}
=
\frac{\lVert v_i\rVert\cos\theta_{ik}+V_i}{2},
\qquad
T^{\mathrm{base}}_{ik}
=
\kappa\frac{r_{ik}}{\max(\varepsilon,A_{ik})}.
$$

GameplayFootball 的离散动作和动画阶段会改变实际到达过程，因此当前实体 token 与附着于该球员的 64 维持久状态共同为基础时间提供正值修正：

$$
T^{\mathrm{eng}}_{ik}
=
T^{\mathrm{base}}_{ik}
\exp\left[
1.5\tanh r_\eta([e_t^i,h_t^i],x_k-p_i)
\right].
$$

球队到达时间使用可微最小值汇总十一名球员。这个运算是普通最小值的平滑近似：最快球员提供主要贡献，其他接近该位置的球员保留较小贡献，从而允许梯度连续传播。双方球队的到达时间为：

$$
\widetilde T_{qk}
=
-0.1\log\sum_{i\in q}
\exp\left(-\frac{T^{\mathrm{eng}}_{ik}}{0.1}\right),
\qquad
q\in\{\mathrm{own},\mathrm{opp}\}.
$$

该到达时间随后生成位置控制程度，其中温度参数 $0.15$ 控制双方到达时间差映射到控制概率时的陡峭程度：

$$
C_t(x_k)
=
\sigma\left(
\frac{\widetilde T_{\mathrm{opp},k}-\widetilde T_{\mathrm{own},k}}
{0.15}
\right).
$$

训练器使用公开观察中的 `time_to_ball_ms` 校准到球位置的预测时间。这个监督把方向相关的运动学先验适配到引擎实际轨迹，同时让 96 个位置 token 使用可达时间权重汇总双方球员的持久状态，并表达双方当前能够控制的区域。

对所有活跃球员，控制校准损失比较预测到球时间与引擎公开值，并用对数压缩较大的时间差。Smooth L1 损失在误差较小时采用二次变化，在误差较大时采用线性变化，从而降低极端时间误差对更新的支配：

$$
\mathcal L_{\mathrm{control}}
=
\mathrm{SmoothL1}\left(
\log(1+10\widehat T_{i,\mathrm{ball}}),
\log(1+10T_{i,\mathrm{ball}})
\right).
$$

因此，解析运动学过程决定可达时间的基础形状，引擎轨迹监督学习动作阶段与连续历史造成的修正，修正后的可达时间再生成控制场和位置 token。

### 4.2 状态相关的位置重要性与空间质量

球队控制某一区域所产生的意义同时取决于控制程度和该区域在当前局面中的重要性。对于网格位置 $x_k$、球位置 $b_t$ 和球速度 $\dot b_t$，共享网络首先为己方进攻方向计算位置重要性：

$$
W_t^{\mathrm{own}}(x_k)
=
\sigma\left(
f_\omega[x_k,x_k-b_t,\dot b_t]
\right)
R^+(x_k),
\qquad
R^+(x_k)=\mathrm{clip}\left(\frac{x_k^{(1)}+1}{2},0,1\right).
$$

对手沿相反方向进攻，因此同一组参数在旋转后的坐标中计算对手位置重要性：

$$
W_t^{\mathrm{opp}}(x_k)
=
\sigma\left(
f_\omega[-x_k,b_t-x_k,-\dot b_t]
\right)
R^-(x_k),
\qquad
R^-(x_k)=\mathrm{clip}\left(\frac{1-x_k^{(1)}}{2},0,1\right).
$$

防守球队通常会集中保护当前局面中的高价值区域，因此其控制分布可以为位置重要性提供监督。Jackaroo 从公开观察计算防守覆盖，并按当前持球方构造监督目标：

$$
\overline W_t^{\mathrm{own}}(x_k)
=
[1-C_t(x_k)]R^+(x_k),
\qquad \text{己方持球},
$$

$$
\overline W_t^{\mathrm{opp}}(x_k)
=
C_t(x_k)R^-(x_k),
\qquad \text{对手持球}.
$$

位置重要性辅助损失只使用具有明确持球方的有效步骤：

$$
\mathcal L_{\mathrm{space}}
=
\mathrm{SmoothL1}\left(
W_t^q(x_k),\overline W_t^q(x_k)
\right).
$$

空间质量把位置重要性与球队控制概率结合：

$$
Q_t^{\mathrm{own}}(x_k)
=
W_t^{\mathrm{own}}(x_k)C_t(x_k),
$$

$$
Q_t^{\mathrm{opp}}(x_k)
=
W_t^{\mathrm{opp}}(x_k)[1-C_t(x_k)].
$$

每个位置 token 同时接收双方空间质量。全局状态使用双方位置重要性之和作为注意力权重，汇总 96 个位置 token：

$$
\alpha_t(x_k)
=
\frac{W_t^{\mathrm{own}}(x_k)+W_t^{\mathrm{opp}}(x_k)}
{\sum_j[W_t^{\mathrm{own}}(x_j)+W_t^{\mathrm{opp}}(x_j)]+\varepsilon},
$$

$$
C_t^{\mathrm{summary}}
=
\sum_k\alpha_t(x_k)e_t^{\mathrm{space}}(x_k).
$$

这个摘要进入全局 GRU，唯一的下一球价值头随后从全局循环状态估计比赛双方取得下一球的相对可能性。位置重要性和空间质量共同构成该价值头的内部空间表示。

### 4.3 动作条件转移

“解析控制场”表示直接由球员位置、速度、方向和疲劳经过第 4.1 节公式计算得到的 96 个控制概率。它提供稳定的公开状态目标，而由循环状态校准的控制表示继续服务于当前策略编码。

“动作条件转移”表示在当前历史和某个候选动作已经给定的条件下，对下一步公开状态、下一步内部状态和进球边界进行预测。动作结果模型同时评价 32 个候选动作，并为每个动作预测下一步全局隐状态、216 维公开结果向量和三类边界概率。公开结果向量由球的位置与速度、二十二名球员的平面位置与速度、96 维解析控制场、球权队伍、持球队员和下一决策节点组成：

$$
z_{t+1}
=
\left[
b_{t+1},
\{p_{t+1}^i,v_{t+1}^i\}_{i=1}^{22},
C_{t+1}^{\mathrm{analytic}},
o_{t+1}^{\mathrm{team}},
o_{t+1}^{\mathrm{player}},
d_{t+1}
\right]\in\mathbb R^{216}.
$$

每项预测都以当前观察历史 $h_t$ 与候选动作 $a_t$ 为条件，因此模型表达的是 $P(s_{t+1}\mid h_t,a_t)$ 所描述的局面相关因果关系。同一组动作在不同局面中会产生不同结果，进球链由连续的真实局面转移构成。

令 $z_{t+1}$ 表示这组公开结果，$d_{t+1}\in\{\mathrm{continue},\mathrm{own\ goal},\mathrm{opponent\ goal}\}$ 表示 segment 边界类别。动作结果头实现：

$$
(\widehat h_{t+1}^a,\widehat z_{t+1}^a,\widehat d_{t+1}^a)
=
f_{\mathrm{transition}}(h_t^{\mathrm{global}},h_t^{d_t},e_a).
$$

因此，这个头共同参数化公开结果、下一全局隐状态和 segment 边界的动作条件预测：

$$
P_\theta\left(
z_{t+1},h_{t+1}^{\mathrm{global}},d_{t+1}
\mid h_t^{\mathrm{global}},h_t^{d_t},a
\right).
$$

已执行动作与真实相邻观察提供监督。连续量使用均方误差，隐状态使用 smooth L1，边界类别使用交叉熵：

$$
\mathcal L_{\mathrm{transition}}
=
\lVert\widehat z_{t+1}-z_{t+1}\rVert_2^2
+
\mathrm{SmoothL1}(\widehat h_{t+1},h_{t+1})
+
\mathrm{CE}(\widehat d_{t+1},d_{t+1}).
$$

这组目标使用实际执行结果训练动作后果表示。真实引擎产生全部 rollout。对于每个候选动作，预测结果中的球位置、球速度和解析控制场还会生成预测空间质量差：

$$
M_t
=
\frac{1}{K}\sum_{k=1}^{K}
\left[
Q_t^{\mathrm{own}}(x_k)-Q_t^{\mathrm{opp}}(x_k)
\right],
$$

$$
\Delta M_t^a
=
\widehat M_{t+1}^a-M_t.
$$

因此，动作头可以区分直接占据高质量空间的动作，以及通过移动球和牵动球员改善后续空间关系的动作。全局 GRU 继续保存跨步骤变化，使多步空间生成过程进入下一球价值状态。

## 5. 单一下一球价值与动作价值

全局循环状态进入唯一的价值头，输出第 1 节定义的下一球价值 $V_\phi(s_t)$。下标 $\phi$ 表示该估计由价值头的可学习参数产生。

这个 $V_\phi$ 同时承担 PPO Critic 和空间控制结果的最终价值函数。Critic 是 PPO 中估计状态价值的分支，它为优势计算提供基线。价值头使用真实 segment 结果进行回归：

$$
\mathcal L_V
=
\left(V_\phi(s_t)-G_k\right)^2.
$$

“动作价值”表示在当前局面选择特定动作后，己方取得下一球的期望结果。动作结果模型为动作 $a$ 产生继续、己方进球和对方进球的概率 $p_c^a,p_+^a,p_-^a$，并预测下一隐状态 $\widehat h_{t+1}^a$。由同一个下一球价值头推导的单步动作价值为：

$$
Q_\phi(s_t,a)
=
p_+^a-p_-^a+p_c^aV_\phi(\widehat h_{t+1}^a).
$$

因此，$Q_\phi$ 通过“当前步骤直接结束 segment 的结果”与“比赛继续后的下一状态价值”组成单步动作价值。状态价值与动作价值共享同一个下一球价值头。已执行动作的 $Q_\phi(s_t,a_t)$ 同样回归真实 $G_k$，从而校准动作结果概率与下一状态价值的组合。

$$
\mathcal L_Q
=
\left(Q_\phi(s_t,a_t)-G_k\right)^2.
$$

Actor 负责为 32 个动作产生相对偏好。它先把全局状态、决策节点的当前 token、该球员的持久状态和空间摘要压缩为关系表示 $r_t^{d_t}$，再为每个候选动作拼接关系表示、64 维动作 embedding、编码后的预测公开结果、预测空间质量 $\widehat M_{t+1}^a$ 及其变化 $\Delta M_t^a$。Actor 输出的 logit 是 softmax 接收的原始分数。分数越高，动作获得的概率越大。一个初始值为零的可学习门把同一动作的 $Q_\phi$ 加入 logit。Actor 使用停止梯度后的预测结果和动作价值，PPO 负责学习策略偏好，真实相邻观察和 segment 结果负责训练转移与价值语义。

对应的动作 logit 为：

$$
\ell_t^a
=
f_{\mathrm{actor}}\left(
r_t^{d_t},e_a,f_z(\widehat z_{t+1}^a),
\widehat M_{t+1}^a,\Delta M_t^a
\right)
+g_Q Q_\phi(s_t,a),
$$

其中 $g_Q$ 是从零开始学习的标量门。softmax 把三十二个有限 logit 转换成总和为 $1$ 的动作概率，策略再从该分布采样 PPO 动作。

$$
\pi_{\mathrm{action}}(a\mid s_t,d_t)
=
\mathrm{softmax}\left(\ell_t^{:}\right)^a,
\qquad
a_t\sim\pi_{\mathrm{action}}(\cdot\mid s_t,d_t).
$$

### 5.1 理论来源

本节的运动学控制过程参考 [A Physics-Driven Study of Dominance Space in Soccer](DSS.pdf)，位置重要性与空间质量关系参考 [Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer](https://www.lukebornn.com/papers/fernandez_ssac_2018.pdf)，下一球价值与动作结果的组合关系参考 [A Framework for the Fine-Grained Evaluation of the Instantaneous Expected Value of Soccer Possessions](EPV.pdf)。Jackaroo 把这些关系统一为上述状态编码、动作条件转移和单一价值头。

## 6. TamakEri 预训练

正式训练在 Jackaroo self-play 之前执行一次预训练。`training/jackaroo/tamakeri.pt` 保存教师的 TorchScript 权重，训练端复现 TamakEri 的观察转换、合法动作过滤、历史状态和两步定向传球协议，并让同一个教师模型同时控制完整比赛的左右双方。`--pretraining-games` 控制完整教师比赛数，`--flight` 控制并发比赛数，默认配置分别为 256 和 16。`--maximum-steps` 控制每场比赛的步数上限，默认值为 3000。采样器按照完整比赛形成数据集。

行为克隆使用教师轨迹中的观察与动作标签监督 Jackaroo 的 Actor，使新策略先获得教师展示的基础动作能力。TamakEri 的一次决策可以把 pending 方向动作继续交给上一步受控球员，而 Jackaroo 的策略动作表示当前 `designated` 球员执行的一个原子动作。因此，采样器使用教师完整十一人动作数组驱动引擎，并为行为克隆设置动作可表示性 mask。教师动作接收者与当前 `designated` 一致时，32 维动作标签进入交叉熵和动作条件转移监督。pending 动作继续作用于原球员时，该 transition 继续参与下一球价值、到达时间校准与空间表示学习。这个处理完整保留教师比赛行为，同时让 Jackaroo 学习自身接口能够表达的动作。

教师比赛中的全部进球 segment 都进入预训练，其中所有可表示动作进入行为克隆。双方轨迹分别取得 $+1$ 与 $-1$，并监督下一球价值和已执行动作的价值。终场 $0:0$ 尾段整体退出预训练数据，因此每条保留轨迹都具有明确的下一球结果。预训练目标为：

$$
\mathcal L_{\mathrm{pre}}
=
\mathcal L_{\mathrm{behavior}}
+c_V\mathcal L_V
+c_T\mathcal L_{\mathrm{transition}}
+c_Q\mathcal L_Q
+c_C\mathcal L_{\mathrm{control}}
+c_S\mathcal L_{\mathrm{space}}.
$$

行为交叉熵给随机初始化策略提供可执行足球动作的先验，真实相邻观察训练动作条件结果，进球 segment 训练下一球价值。完成指定 epoch 后，训练器保存包含预训练网络、优化器和统计信息的 checkpoint，再从同一 seed 序列的后续比赛开始 Jackaroo self-play PPO。随后的策略更新全部使用 Jackaroo 自己生成的引擎轨迹。恢复已有 checkpoint 时直接继续 PPO。

## 7. PPO 与数据流

采样器让 C++ 向量环境以 flight 形式并行推进完整比赛，并在每个决策步把所有活跃比赛的左右观察合并为一次 GPU 推理。正式脚本默认每个 flight 并行 16 场比赛，并在一次更新中完成 256 场比赛。`--flight` 与 `--games-per-update` 可以分别调整这两个数量。每场比赛最多推进 3000 个引擎步，双方在每个有效步骤各自产生一次策略决策。进球会同时结束两侧的当前 segment，并产生结果互为相反数的两条训练轨迹。比赛结束时的尾段结果记为 $0$，PPO 数据集保留结果为 $+1$ 或 $-1$ 的 segment。一次更新保留的进球 segment 数由该批完整比赛中的实际进球数决定。

对有效 transition，优势表示实际 segment 结果相对当前价值估计高出或低出的程度：

$$
\widehat A_t=G_k-V_\phi(s_t).
$$

PPO 使用旧策略记录的 log probability 计算概率比，并采用标准 clipped objective：

$$
\log p_t
=
\log\pi_{\mathrm{action}}(a_t\mid s_t,d_t).
$$

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

最终训练目标把策略、单一下一球价值、动作结果、动作价值、控制场校准和空间表示合并为：

$$
\mathcal L
=
-\mathcal J_{\mathrm{clip}}
+c_V\mathcal L_V
-c_H\mathcal H(\pi_\theta)
+c_T\mathcal L_{\mathrm{transition}}
+c_Q\mathcal L_Q
+c_C\mathcal L_{\mathrm{control}}
+c_S\mathcal L_{\mathrm{space}}.
$$

近似 KL 衡量新旧策略概率分布的平均变化，clipping fraction 表示触发 PPO 比率裁剪的样本比例，两者共同反映一次更新的幅度。每次更新还记录策略损失、下一球价值损失、策略熵、转移损失、动作价值损失、控制场损失、空间表示损失、动作分布、进球 segment 数、删失尾段数量、完整自对弈比分、每秒引擎步数、每秒 PPO transition 数和峰值 GPU 显存。策略熵衡量动作分布的分散程度，用于观察探索是否过早消失。控制台摘要、JSONL 日志和 checkpoint 诊断字段共同保存这些数据。

## 8. 训练与评估

正式训练先完成第 6 节的教师预训练，再开始 Jackaroo self-play。训练 seed 初始化一个可复现的伪随机序列，教师比赛与 Jackaroo 比赛依次从该序列取得唯一的 32 位引擎 seed。一个共享 Jackaroo 网络同时扮演 self-play 的两侧，使每粒进球都提供正负成对的策略样本，并让双方动作都成为可学习决策。每次 PPO 更新完成后，策略使用固定 seed 42 与内置 AI 进行两场完整验证，Jackaroo 分别位于左右物理侧。验证采用贪心动作、记录整场比分，并把两场比赛写入同一个 GFR 文件。验证轨迹专门用于整场比分与录像评估，训练数据来自 self-play rollout。

检查点保存网络、优化器、网络结构标识、观察与动作维度、序列长度、训练 seed、损失权重、更新编号和最新诊断。恢复训练时，加载器校验这些决定张量语义的字段，然后继续更新。

训练结束后，导出器把策略编译为 TorchScript。部署接口接收一个公开观察、24 个实体循环状态和全局循环状态，并返回 32 个动作 logit、下一球价值及更新后的循环状态。
