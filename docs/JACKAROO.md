# Jackaroo 实现说明

## 1. 训练目标

Jackaroo 把完整足球比赛分解为连续的进球 segment。第一个 segment 从比赛开球开始，此后每粒进球同时结束当前 segment 并开始下一次开球。己方率先取得下一球时，segment 结果为 $+1$。对方率先取得下一球时，结果为 $-1$。比赛到达时间上限时，最后一个 segment 形成结果未知的尾段。这个尾段中的相邻状态、教师动作、到达时间和空间状态继续提供局部监督。下一球策略、状态价值和动作价值只使用结果为 $+1$ 或 $-1$ 的 segment。

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

Jackaroo 采用单智能体控制。训练采用共享策略自对弈，同一个 Jackaroo 网络分别接收左右两队的规范化观察，并在每一步为双方各自的决策节点 $d_t$ 选择一个原子动作。每队其余球员继续由 TeamAI 与 Eliza 驱动。引擎在收齐双方决策后统一执行完整足球规则、物理、动画和球队行为，并分别返回两侧的下一步公开观察。

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

观察编码把 $d_t$ 保存为当前队伍的 anchor。anchor 表示本次动作的接收球员。Entity Transformer 在第 $d_t$ 个己方球员 token 上加入可学习的 anchor 标记，并从该 token 读取球员的位置、速度、朝向、角色、疲劳、动作阶段及其相对于球和其他球员的关系。Actor 同时使用这名球员的实体表示、附着于该球员的 GRU 状态和完整局面状态，因此动作分布为：

$$
\pi_\theta(a_t\mid s_t,d_t).
$$

一个训练样本由 $(s_t,d_t,a_t)$ 构成，$d_t$ 同时参与动作推理与动作提交。策略从观察 $(s_t,d_t)$ 产生 $a_t$，引擎随后把 $a_t$ 写入第 $d_t$ 名球员的动作位置。

令 $a_t^{\mathrm{delegate}}$ 表示把一名球员的动作决定权交给 TeamAI 与 Eliza。引擎接收的十一人联合动作为：

$$
A_t^i=
\begin{cases}
a_t, & i=d_t,\\
a_t^{\mathrm{delegate}}, & i\ne d_t.
\end{cases}
$$

因此，Jackaroo 每一步直接决定当前节点的动作，其余十名球员继续由 TeamAI 与 Eliza 驱动。

固定的引擎决策节点把训练样本集中在持球、接球、争球和关键防守等高影响局面。Jackaroo 在这些位置承担动作结果，并通过进球 segment 回报学习处理球和改变球队空间关系。

每个动作使用 18 维固定结构特征。它由二维方向向量 $u_a$、13 维功能类别 $c_a$ 和 3 维空闲、按下、释放阶段 $q_a$ 拼接而成，所以 $18=2+13+3$。13 个功能类别对应空闲、方向和十一个足球功能。动作 0 为空闲，动作 1 至 8 是八个方向的按下，动作 9 至 19 是十一个足球功能的按下，动作 20 是方向释放，动作 21 至 31 是对应足球功能的释放。动作编码器把这组特征映射为 64 维 embedding：

$$
x_a=[u_a,c_a,q_a]\in\mathbb R^{18},
\qquad
e_a=f_{\mathrm{action}}(x_a)\in\mathbb R^{64}.
$$

这里的动作 embedding 是由可学习映射产生的固定宽度动作向量。八个移动方向通过 $u_a$ 表达方向邻近关系，足球功能通过 $c_a$ 区分，持续动作的按下与释放共享功能类别并由 $q_a$ 区分阶段。该表示让相邻方向、相同足球功能和成对的按下/释放动作共享统计规律。

## 3. 持久实体策略网络

Entity Transformer 首先在每个决策步建模二十二名球员、球和比赛上下文之间的关系。它采用三个 8 头 self-attention block。self-attention 根据当前状态学习每对实体之间的权重，使每个实体表示吸收与其决策相关的其他实体信息。Entity Transformer 最终产生 128 维实体表示，并按照 anchor 索引提取当前受控球员的表示，再汇总双方球队、球和比赛上下文。

随后，每个实体使用共享参数的 GRUCell 更新自己的 64 维持久状态，完整局面使用另一个 GRUCell 更新 256 维全局状态。GRUCell 是带门控的循环单元，它把当前输入与上一步状态结合，并通过可学习的门选择进入新状态的历史信息。实体状态始终附着于同一名球员，因此决策节点从球员 A 变为球员 B 时，双方的跑动趋势、动作进度和历史关系仍保持各自身份。全局状态保存跨实体的局面历史。每个进球 segment 结束时，训练器清空该环境的全部循环状态。

二十二名球员、球和比赛上下文各自保存一个 64 维实体状态。二十四个实体共享同一个 Entity GRUCell 参数。状态为 inactive 的球员更新结果归零。Global GRUCell 的输入同时汇合扁平公开观察、Entity Transformer 关系摘要和空间网格摘要：

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
+f_{\mathrm{space}}(S_t^{\mathrm{space}}),
$$

$$
h_t^{\mathrm{global}}
=
\mathrm{GRU}_{\mathrm{global}}\left(
u_t,
h_{t-1}^{\mathrm{global}}
\right),
$$

其中 $m_t^i$ 是实体有效性 mask。mask 是取值为 $0$ 或 $1$ 的指示量，状态为 inactive 的球员取得 $0$。$E_t$ 是 Entity Transformer 关系摘要。$S_t^{\mathrm{space}}$ 是由 96 个球场位置 token 加权汇总得到的空间网格摘要。位置 token 是表示一个固定网格位置的向量，其中包含该位置的控制关系、空间质量和最可能到达该处的球员状态。当前决策节点的实体 token、持久实体状态、全局状态和空间网格摘要随后组成 Actor 的关系表示。第 4 节给出位置 token、权重和汇总过程的完整定义。

PPO 使用长度为 $L$ 的连续学习窗口执行 truncated backpropagation through time，当前取 $L=32$。每个学习窗口前面带有 $L_{\mathrm{burn}}=32$ 步 burn-in 历史。burn-in 表示用当前网络重新计算循环状态的历史区间，这个区间关闭梯度，后面的学习窗口正常参与反向传播。第一个窗口缺少的历史位置使用 mask 为 $0$ 的 padding，后续窗口从前一个 32 步块的起点状态开始重放历史。每个序列保存该历史起点的循环状态 $(H_{t_0-1},h_{t_0-1}^{\mathrm{global}})$ 和有效步骤 mask。序列始终位于同一个进球 segment 内。进球后两侧的实体状态和全局状态同时归零。因此，Actor 和 Critic 在更新时看到由当前参数重建的近期上下文，梯度长度保持为 32 步，正向隐藏状态仍沿完整 segment 连续传递。

## 4. 空间控制与动作结果

令“空间建模数据流”表示从公开球场状态形成空间表示，再由候选动作预测空间变化，最终估计下一球价值的完整计算过程。它依次经过空间表示模块、动作条件转移模块、Global GRUCell 和价值头。

空间表示模块记为 $f_{\mathrm{space}}$。它接收双方球员状态、球状态、实体 token 和持久实体状态，并在固定网格上计算球员到达时间、球队控制概率、位置重要性、空间质量和位置 token。球场坐标把纵向范围规范化为 $[-1,1]$，并在相同比例下把横向范围表示为 $[-0.42,0.42]$。网格由两个包含端点的均匀序列构成：

$$
X=\left\{-1+\frac{2j}{11}\mid j=0,\ldots,11\right\},
\qquad
Y=\left\{-0.42+\frac{0.84l}{7}\mid l=0,\ldots,7\right\}.
$$

全部位置是两个序列的笛卡尔积：

$$
\mathcal G=X\times Y,
\qquad
K=|\mathcal G|=12\times8=96.
$$

因此，96 个位置形成覆盖整个规范化球场的均匀矩形点阵。纵向相邻点间隔为 $2/11$，横向相邻点间隔为 $0.84/7=0.12$，球场边界也属于点阵。$12\times8$ 是空间表示的内部离散化常量，它固定位置 token 的数量以及后续张量的维度。该模块输出的位置 token 经过加权汇总形成空间网格摘要 $S_t^{\mathrm{space}}$。

“到达时间”表示一名球员从当前的位置、速度和朝向出发，在当前疲劳状态下到达指定球场位置所需的归一化模拟时间。较小的到达时间表示该球员能更早影响该位置。

“球队控制概率”表示双方竞争同一位置时，己方先到达并控制该位置的相对程度，记为 $C_t(x_k)\in[0,1]$。数值接近 $1$ 表示己方占优，接近 $0$ 表示对手占优，接近 $0.5$ 表示双方到达能力接近。

“位置重要性”表示控制一个位置对当前进攻方向和球状态的相关程度，记为 $W_t^q(x_k)\in[0,1]$，其中 $q$ 表示己方或对手。它随球的位置、球的速度和球队进攻方向变化。

“空间质量”表示一支球队对重要位置实际拥有的控制量。己方空间质量等于己方位置重要性乘以己方控制概率。对手空间质量等于对手位置重要性乘以对手控制概率。这个乘积使网络把参数集中在同时具备局面意义和实际可控性的区域。

完整的空间建模数据流把受控球员动作、其他球员响应、控制关系变化、空间质量变化和下一球价值连接为一条可训练因果链：

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

空间表示模块首先估计每名球员对每个网格位置的到达时间，再把双方到达时间转换成连续的球队控制概率。状态相关的位置重要性与控制概率相乘后形成空间质量。空间网格摘要进入 Global GRUCell，动作条件转移模块预测各候选动作造成的空间变化，价值头估计第 1 节定义的下一球价值。公开引擎轨迹提供到达时间和位置重要性的校准目标，真实进球 segment 提供下一球价值的监督目标。

### 4.1 方向相关的到达时间与球队控制

空间表示模块在 $K=96$ 个球场位置上估计双方球员的可达性。对于球员 $i$ 和位置 $x_k$，该模块根据当前位置 $p_i$、速度 $v_i$、运动方向和疲劳得到基础到达时间。令 $f_i\in[0,1]$ 表示公开观察中的疲劳程度，并定义疲劳调整后的速度先验：

$$
V_i=V_{\max}(1-\lambda_f f_i).
$$

当前取 $V_{\max}=0.012$ 和 $\lambda_f=0.35$。$V_{\max}$ 表示规范化球场坐标中的每步速度尺度，$\lambda_f$ 表示疲劳从最低值变化到最高值时该尺度的相对衰减上限。后续的可学习修正负责适配引擎中的动作阶段与实际轨迹。令 $\kappa=10^{-3}$ 把位置距离与速度之比缩放到到球时间监督使用的数值范围：

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
\lambda_r\tanh r_\eta([e_t^i,h_t^i],x_k-p_i)
\right].
$$

当前取修正幅度 $\lambda_r=1.5$。由于 $\tanh$ 的值位于 $[-1,1]$，乘性修正被限制在 $[e^{-1.5},e^{1.5}]$ 内。这个范围允许历史状态显著调整解析估计，同时保持到达时间为正数。

球队到达时间使用可微最小值汇总十一名球员。这个运算是普通最小值的平滑近似。最快球员提供主要贡献，其他接近该位置的球员保留较小贡献，从而允许梯度连续传播。令 $\tau_T$ 表示可微最小值的温度：

$$
\widetilde T_{qk}
=
-\tau_T\log\sum_{i\in q}
\exp\left(-\frac{T^{\mathrm{eng}}_{ik}}{\tau_T}\right),
\qquad
q\in\{\mathrm{own},\mathrm{opp}\}.
$$

当前取 $\tau_T=0.1$。较小的温度使结果接近最快球员的到达时间，正温度同时保留其他接近最快值球员的梯度。

空间表示模块随后把双方球队的到达时间差映射为球队控制概率。令 $\tau_C$ 表示概率映射的温度：

$$
C_t(x_k)
=
\sigma\left(
\frac{\widetilde T_{\mathrm{opp},k}-\widetilde T_{\mathrm{own},k}}
{\tau_C}
\right).
$$

当前取 $\tau_C=0.15$。该值决定到达时间差转化为控制优势的陡峭程度。较小的值形成接近胜负判定的控制边界，较大的值形成更平滑的竞争区域。

公开观察包含每名球员到达球位置所需的模拟时间 $T_{i,\mathrm{ball}}$。训练器用这个量校准预测时间。该监督把方向相关的运动学先验适配到引擎实际轨迹，同时让 96 个位置 token 使用可达时间权重汇总双方球员的持久状态，并表达双方当前能够控制的区域。

对所有活跃球员，控制校准损失比较预测到球时间与引擎公开值，并用对数压缩较大的时间差。令 $s_T=10$ 表示进入对数前的时间尺度，它把经过 $\kappa$ 归一化的时间扩展到便于优化的数值范围。Smooth L1 损失在误差较小时采用二次变化，在误差较大时采用线性变化，从而降低极端时间误差对更新的支配：

$$
\mathcal L_{\mathrm{control}}
=
\mathrm{SmoothL1}\left(
\log(1+s_T\widehat T_{i,\mathrm{ball}}),
\log(1+s_TT_{i,\mathrm{ball}})
\right).
$$

因此，解析运动学过程决定可达时间的基础形状。引擎轨迹为动作阶段与连续历史造成的修正提供监督。空间表示模块再用修正后的可达时间计算控制概率和位置 token。

### 4.2 状态相关的位置重要性与空间质量

球队控制某一区域所产生的意义同时取决于控制程度和该区域在当前局面中的重要性。共享参数的位置重要性函数 $f_\omega$ 接收网格位置 $x_k$、球位置 $b_t$ 和球速度 $\dot b_t$，并为己方进攻方向计算位置重要性：

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

训练器只在持球方明确的有效步骤上计算位置重要性损失：

$$
\mathcal L_{\mathrm{space}}
=
\mathrm{SmoothL1}\left(
W_t^q(x_k),\overline W_t^q(x_k)
\right).
$$

Jackaroo 把位置重要性与球队控制概率相乘，并把结果定义为空间质量：

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

空间表示模块把双方空间质量写入每个位置 token。策略网络使用双方位置重要性之和作为注意力权重，并汇总 96 个位置 token：

$$
\alpha_t(x_k)
=
\frac{W_t^{\mathrm{own}}(x_k)+W_t^{\mathrm{opp}}(x_k)}
{\sum_j[W_t^{\mathrm{own}}(x_j)+W_t^{\mathrm{opp}}(x_j)]+\varepsilon},
$$

$$
S_t^{\mathrm{space}}
=
\sum_k\alpha_t(x_k)e_t^{\mathrm{space}}(x_k).
$$

空间网格摘要 $S_t^{\mathrm{space}}$ 进入 Global GRUCell。唯一的下一球价值头随后从全局循环状态估计比赛双方取得下一球的相对可能性。位置重要性和空间质量共同构成该价值头的内部空间表示。

### 4.3 动作条件转移

“解析控制场”表示直接由球员位置、速度、方向和疲劳经过第 4.1 节公式计算得到的 96 个控制概率。它提供稳定的公开状态目标，而由循环状态校准的控制表示继续服务于当前策略编码。

“动作条件转移”表示在当前历史和某个候选动作已经给定的条件下，对下一步公开状态、下一步内部状态和进球边界进行预测。动作条件转移模型记为 $f_{\mathrm{transition}}$。它同时评价由环境定义的 32 个候选动作，并为每个动作预测下一步全局隐状态、公开结果向量和三类边界概率。三类边界分别表示 segment 继续、己方进球和对方进球。

公开结果向量的 216 个分量均由可观测量构成。球的三维位置和三维速度提供 6 个分量。二十二名球员各自的二维位置和二维速度提供 $22\times4=88$ 个分量。解析控制场提供 $K=96$ 个分量。球权队伍在无归属、己方和对方之间形成 3 维 one-hot 表示。持球队员在无归属和十一名球员之间形成 12 维 one-hot 表示。下一决策节点形成 11 维 one-hot 表示。因此总维度为：

$$
6+22\times4+96+3+12+11=216.
$$

将这些分量写成向量可得：

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

令 $z_{t+1}$ 表示这组公开结果，$y_{t+1}\in\{\mathrm{continue},\mathrm{own\ goal},\mathrm{opponent\ goal}\}$ 表示 segment 边界类别。动作条件转移模型实现：

$$
(\widehat h_{t+1}^a,\widehat z_{t+1}^a,\widehat y_{t+1}^a)
=
f_{\mathrm{transition}}(h_t^{\mathrm{global}},h_t^{d_t},e_a).
$$

因此，这个头共同参数化公开结果、下一全局隐状态和 segment 边界的动作条件预测：

$$
P_\theta\left(
z_{t+1},h_{t+1}^{\mathrm{global}},y_{t+1}
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
\mathrm{CE}(\widehat y_{t+1},y_{t+1}).
$$

这组目标使用实际执行结果训练动作后果表示。真实引擎产生全部训练轨迹。对于每个候选动作，预测结果中的球位置、球速度和解析控制场还会生成预测空间质量差：

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

Actor 接收每个候选动作的预测空间质量及其变化，因此可以学习区分直接占据高质量空间的动作，以及通过移动球和牵动球员改善后续空间关系的动作。Global GRUCell 保存跨步骤变化，使多步空间生成过程进入下一球价值状态。

## 5. 单一下一球价值与动作价值

全局循环状态进入唯一的价值头，输出第 1 节定义的下一球价值 $V_\phi(s_t)$。下标 $\phi$ 表示该估计由价值头的可学习参数产生。

这个 $V_\phi$ 同时承担 PPO Critic 和空间控制结果的最终价值函数。Critic 是 PPO 中估计状态价值的分支，它为优势计算提供基线。价值头使用真实 segment 结果进行回归：

$$
\mathcal L_V
=
\left(V_\phi(s_t)-G_k\right)^2.
$$

“动作价值”表示在当前局面选择特定动作后，己方取得下一球的期望结果。动作条件转移模型为动作 $a$ 产生继续、己方进球和对方进球的概率 $p_c^a,p_+^a,p_-^a$，并预测下一隐状态 $\widehat h_{t+1}^a$。同一个下一球价值头据此计算单步动作价值：

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

预处理程序复现 TamakEri 的观察转换、合法动作过滤、历史状态和两步定向传球协议，并让同一个教师策略同时控制完整比赛的左右双方。它把公开观察、下一步公开观察、教师动作、动作可表示性、segment 边界和 segment 结果转换成定长连续序列，并使用 gzip 压缩写入 HDF5 文件。这个过程只执行轨迹生成。生成的数据文件可以反复用于不同 Jackaroo 初始化、批量大小和预训练轮数。

预训练程序从 HDF5 文件中随机排列完整球队轨迹，在每条轨迹内部依次读取定长序列并送入 GPU。该阶段的数据流由 HDF5、Jackaroo 网络和优化器构成。引擎采样与教师推理已经在预处理阶段完成。令 $N_{\mathrm{teacher}}$ 表示教师自对弈比赛数，$B_{\mathrm{parallel}}$ 表示预处理时同时推进的比赛数，$H$ 表示一场完整比赛的最大决策步数。当前数据生成配置取：

$$
N_{\mathrm{teacher}}=256,
\qquad
B_{\mathrm{parallel}}=16,
\qquad
H=3000.
$$

$H=3000$ 对应环境采用的完整比赛时域。$B_{\mathrm{parallel}}=16$ 让一次批量推理同时服务多场比赛。$N_{\mathrm{teacher}}=256$ 提供十六批完整比赛，使教师数据集包含多个随机初态和进球 segment。

行为克隆使用教师轨迹中的观察与动作标签监督 Jackaroo 的 Actor，使新策略先获得教师展示的基础动作能力。TamakEri 的一次决策可以把两步动作中的第二个方向动作继续交给上一步受控球员，而 Jackaroo 的策略动作表示当前决策节点执行的一个原子动作。因此，采样器使用教师完整十一人联合动作驱动引擎，并以可表示性 mask $m_t^{\mathrm{BC}}$ 标记当前决策节点能够执行的教师动作。教师动作接收者与当前决策节点一致时，$m_t^{\mathrm{BC}}=1$，对应的 32 维动作标签进入行为交叉熵和动作条件转移监督。第二个方向动作继续作用于原球员时，$m_t^{\mathrm{BC}}=0$，对应状态继续参与下一球价值、到达时间校准与空间表示学习。行为克隆损失为：

$$
\mathcal L_{\mathrm{behavior}}
=
-\frac{
\sum_t m_t^{\mathrm{BC}}\log\pi_{\mathrm{action}}(a_t^{\mathrm{teacher}}\mid s_t,d_t)
}{
\sum_t m_t^{\mathrm{BC}}
}.
$$

教师比赛中的全部进球 segment 都进入预训练，其中所有可表示动作进入行为克隆。双方轨迹分别取得 $+1$ 与 $-1$，并监督下一球价值和已执行动作的价值。结果未知的尾段同样保存到 HDF5，其中可表示动作继续训练行为分布，相邻状态继续训练动作条件转移，到达时间与空间状态继续训练空间表示。下一球策略、状态价值和动作价值的监督集合由取得明确下一球结果的 segment 构成。进球边界分类的监督集合包含实际发生后继状态的步骤。预训练目标为：

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

行为交叉熵给随机初始化策略提供可执行足球动作的先验，真实相邻观察训练动作条件结果，进球 segment 训练下一球价值。当前配置对教师数据执行两轮优化。两轮优化让每条教师样本参与两次梯度更新，同时保留完整数据集带来的状态多样性。HDF5 读取器按照 GPU batch 大小加载压缩序列，因此数据生成速度和引擎内存不再限制预训练 batch。预训练完成后，Jackaroo self-play PPO 使用自身策略生成的新轨迹继续学习。

## 7. PPO 与数据流

异步向量环境同时维护 $B_{\mathrm{parallel}}=16$ 场完整比赛。每个 native 环境在独立工作线程中执行引擎 step。主线程等待首批完成的环境，并把当时已经就绪的左右观察合并为下一次策略推理批次。完成较早的环境可以直接进入下一步，单步耗时较长的环境继续在自己的工作线程中运行。引擎 reset 使用进程级随机状态，因此 reset 在当前已经提交的 step 全部结束后串行执行。

每次 PPO 更新采集 $N_{\mathrm{self}}=256$ 场比赛。一次 rollout 期间的策略参数保持冻结，全部比赛完成后才执行模型更新，因此每条 trajectory 都来自同一个旧策略。每场比赛最多推进 $H=3000$ 个引擎步，双方在每个有效步骤各自产生一次策略决策。进球会同时结束两侧的当前 segment，并产生结果互为相反数的两条训练轨迹。结果未知的尾段继续训练动作条件转移、到达时间和空间表示。PPO policy、下一球 Critic 和动作价值目标使用独立 mask 选择结果为 $+1$ 或 $-1$ 的 segment。一次更新获得的进球 segment 数由该批完整比赛中的实际进球数决定。

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

Critic value clipping 使用采样时记录的旧价值 $V_{\mathrm{old}}(s_t)$ 约束一次 PPO 更新中的价值变化。令 $\epsilon_V=0.2$，裁剪后的候选值为：

$$
V_{\mathrm{clip}}(s_t)
=
V_{\mathrm{old}}(s_t)
+
\mathrm{clip}\left(
V_\phi(s_t)-V_{\mathrm{old}}(s_t),
-\epsilon_V,
\epsilon_V
\right).
$$

Critic 对原始价值误差与裁剪价值误差取较大者：

$$
\mathcal L_V^{\mathrm{clip}}
=
\mathbb E_t\left[
\max\left(
\left(V_\phi(s_t)-G_k\right)^2,
\left(V_{\mathrm{clip}}(s_t)-G_k\right)^2
\right)
\right].
$$

一次模型更新首先执行一轮足球结构学习阶段。这个阶段使用真实相邻观察、segment 结果、引擎公开到球时间和空间控制目标训练动作结果、动作价值、到达时间与空间表示：

$$
\mathcal L_{\mathrm{structure}}
=
c_T\mathcal L_{\mathrm{transition}}
+c_Q\mathcal L_Q
+c_C\mathcal L_{\mathrm{control}}
+c_S\mathcal L_{\mathrm{space}}.
$$

足球结构学习完成后执行四轮 PPO 阶段。PPO 阶段使用最多 2048 个有效 transition 组成一个 mini-batch。由于每条学习序列包含 32 个有效 transition，一个完整 mini-batch 通常包含 64 条序列及其 burn-in 历史。该阶段的损失为：

$$
\mathcal L_{\mathrm{PPO}}
=
-\mathcal J_{\mathrm{clip}}
+c_V\mathcal L_V^{\mathrm{clip}}
-c_H\mathcal H(\pi_\theta)
.
$$

两个阶段共享 Entity Transformer、实体 GRU、Global GRUCell、空间控制模块和 EPV 价值头。因此，足球结构学习产生的参数更新会立即进入 Critic 的局面表征与价值估计，随后执行的 PPO 阶段在具有进球结果的 segment 上使用 clipping 目标继续优化 Actor 和 Critic。把 PPO 放在最后可以避免足球结构学习在 clipping 之后再次修改共享模型。PPO 结束后的完整模型直接用于本轮验证，并在下一轮 rollout 中重新记录新的动作概率和状态价值。

PPO 阶段内部的近似 KL、policy clipping fraction 和 value clipping fraction 描述各个 mini-batch 更新时的变化。KL 在训练过程中只承担观测指标的职责，PPO clipping 为具有进球结果的样本提供稳定更新目标。策略熵衡量动作分布的分散程度。训练评估统计各项损失、动作分布、进球 segment 数、完整自对弈比分和固定种子完整比赛结果。

## 8. 训练与评估

`prepare`、`pretrain` 和 `train` 是三个独立程序。运行脚本按顺序调用它们，并在压缩数据集或预训练 checkpoint 已经存在时复用相应产物。`prepare` 把每支球队在一个进球 segment 内的轨迹写成带有稳定轨迹编号的有序 32 步块。`pretrain` 随机排列完整轨迹，在每条轨迹内部依次读取这些块，并把前一块末尾的实体 GRU 状态和全局 GRU 状态传给后一块。块边界截断梯度，而循环状态继续保存因果历史。`train` 只读取预训练 checkpoint 并执行 Jackaroo self-play PPO。一个初始随机种子生成可复现的比赛种子序列，每场教师比赛与 Jackaroo 比赛取得各自的环境随机种子。一个共享 Jackaroo 网络同时扮演 self-play 的两侧，使每粒进球都提供正负成对的策略样本，并让双方动作都成为可学习决策。

每次 PPO 更新完成后，策略与内置 AI 进行一组换边验证。该组验证由两场使用相同环境随机种子的完整比赛构成，Jackaroo 分别位于左右物理侧。两场比赛的设计消除固定物理侧对评估结果的影响。验证采用贪心动作并记录整场比分。训练数据来自 self-play 轨迹，验证轨迹用于衡量整场竞技表现。
