# ODA-Conditioned Learned Potential

## 1. 目标

ODA-conditioned learned potential 为 Jackaroo 构造一个由比赛结果监督的密集训练信号。该方法采用内置 AI 比赛学习攻防注意力表示，采用完整比赛的胜平负结果学习状态势能，并通过势能差为 PPO 提供逐步反馈。

方法包含三个相互独立的语义层次：

1. ODA 表示球员之间的攻防相关性。
2. 结果价值模型根据真实终场结果判断局面价值。
3. 势能差将局面价值转换成保持任务目标的逐步训练信号。

Jackaroo 的任务目标采用终场胜平负。训练过程将进攻方式、防守方式、传球选择和控球节奏交给策略从比赛结果中学习。

## 2. 决策过程

每个决策步对应 100 ms 引擎时间。公共观察记为：

$$
o_t = \left(
b_t,
P_t^{L},
P_t^{R},
u_t^{L},
u_t^{R},
g_t,
m_t,
\tau_t
\right).
$$

各符号含义如下：

- $b_t$ 表示球的位置、速度和旋转。
- $P_t^L$ 与 $P_t^R$ 表示左右球队各 11 名球员的状态。
- $u_t^L$ 与 $u_t^R$ 表示两队的球权、到球时间和指定控球球员等球队状态。
- $g_t$ 表示当前比分。
- $m_t$ 表示比赛模式、定位球归属和最近触球信息。
- $\tau_t$ 表示比赛时间和决策步数。

训练器将左右球队统一变换到己方从左向右进攻的规范坐标系。镜像变换记为：

$$
\mathcal{M}: o_t \mapsto \widetilde{o}_t.
$$

规范坐标系使同一种足球局面共享同一种表示。训练器同时保留时间、比分、比赛模式和球员角色，因为这些变量共同决定当前局面的策略含义。

策略使用长度为 $L$ 的观察历史：

$$
x_t = \left(o_{t-L+1}, \ldots, o_t\right).
$$

历史序列为动作阶段、传球飞行过程和短期攻防转换提供上下文。

## 3. 数据生成

### 3.1 比赛轨迹

数据生成器在后台模式运行内置 AI 对局，并在每个决策步记录公共观察。单场轨迹记为：

$$
\mathcal{T}
=
\left(o_0, o_1, \ldots, o_T, y_T\right).
$$

终场标签采用胜、平、负三分类：

$$
y_T =
\begin{cases}
(1,0,0), & g_T^{\mathrm{own}} > g_T^{\mathrm{opp}}, \\
(0,1,0), & g_T^{\mathrm{own}} = g_T^{\mathrm{opp}}, \\
(0,0,1), & g_T^{\mathrm{own}} < g_T^{\mathrm{opp}}.
\end{cases}
$$

每场比赛从左右两队视角各产生一条规范化轨迹。该处理扩大样本规模，并显式表达足球比赛的左右对称性。

### 3.2 进攻注意力标签

进攻样本从己方稳定持球时刻开始。当前持球球员记为 $i_t$。数据生成器在长度为 $H_{\mathrm{ODA}}$ 的前向窗口内查找下一名稳定持球球员。目标球员记为：

$$
j_t^{\mathrm{off}} \in \{0,1,\ldots,10\}.
$$

同一球员继续持球时，标签指向当前球员。队友形成稳定接球时，标签指向该队友。样本筛选器接纳球权归属清晰、目标球员有效且比赛处于正常推进阶段的片段。

进攻标签采用 one-hot 向量：

$$
y_{t,k}^{\mathrm{off}}
=
\mathbb{1}\!\left[k=j_t^{\mathrm{off}}\right].
$$

### 3.3 防守注意力标签

防守样本从对方稳定持球时刻开始。数据生成器在同一前向窗口内确定当前或下一名稳定持球的对方球员：

$$
j_t^{\mathrm{def}} \in \{0,1,\ldots,10\}.
$$

防守标签同样采用 one-hot 向量：

$$
y_{t,k}^{\mathrm{def}}
=
\mathbb{1}\!\left[k=j_t^{\mathrm{def}}\right].
$$

攻防标签描述实际比赛轨迹中的球权关联。标签构造只使用公共观察中的球权、最近触球、指定球员、球员状态和时间序列。

## 4. ODA 表示模型

### 4.1 实体编码

每名球员的原始特征记为 $p_{t,i}^{q}$，其中 $q$ 表示己方或对方。共享实体编码器将球员映射到 $d$ 维表示：

$$
h_{t,i}^{q}
=
f_{\theta_{\mathrm{entity}}}\!\left(p_{t,i}^{q}\right).
$$

球和比赛上下文分别形成球实体与比赛实体：

$$
h_t^{\mathrm{ball}}
=
f_{\theta_{\mathrm{ball}}}\!\left(b_t\right),
$$

$$
h_t^{\mathrm{match}}
=
f_{\theta_{\mathrm{match}}}\!\left(g_t,m_t,\tau_t\right).
$$

实体集合写为：

$$
E_t
=
\left\{
h_{t,0}^{\mathrm{own}},\ldots,h_{t,10}^{\mathrm{own}},
h_{t,0}^{\mathrm{opp}},\ldots,h_{t,10}^{\mathrm{opp}},
h_t^{\mathrm{ball}},
h_t^{\mathrm{match}}
\right\}.
$$

### 4.2 对手注意力

对手注意力描述每名己方球员与全部对方球员之间的关系。查询、键和值定义为：

$$
q_{t,i}=W_Q h_{t,i}^{\mathrm{own}},
\qquad
k_{t,j}=W_K h_{t,j}^{\mathrm{opp}},
\qquad
v_{t,j}=W_V h_{t,j}^{\mathrm{opp}}.
$$

注意力权重为：

$$
\alpha_{t,ij}
=
\frac{
\exp\!\left(q_{t,i}^{\mathsf{T}}k_{t,j}/\sqrt{d}\right)
}{
\sum_{r=0}^{10}
\exp\!\left(q_{t,i}^{\mathsf{T}}k_{t,r}/\sqrt{d}\right)
}.
$$

对手上下文为：

$$
c_{t,i}^{\mathrm{opp}}
=
\sum_{j=0}^{10}\alpha_{t,ij}v_{t,j}.
$$

### 4.3 队友注意力

己方球员表示与对手上下文连接后形成关系实体：

$$
z_{t,i}
=
f_{\theta_{\mathrm{relation}}}
\!\left(
h_{t,i}^{\mathrm{own}},
c_{t,i}^{\mathrm{opp}},
h_t^{\mathrm{ball}},
h_t^{\mathrm{match}}
\right).
$$

当前指定球员的关系实体作为查询，11 名己方球员作为候选目标。进攻注意力分布为：

$$
p_t^{\mathrm{off}}
=
\mathrm{softmax}
\!\left(
W_{\mathrm{off}}
\left[z_{t,0},\ldots,z_{t,10}\right]
\right).
$$

防守分支交换攻守实体角色，并输出对方 11 名球员上的分布：

$$
p_t^{\mathrm{def}}
=
\mathrm{softmax}
\!\left(
W_{\mathrm{def}}
\left[\widetilde{z}_{t,0},\ldots,\widetilde{z}_{t,10}\right]
\right).
$$

### 4.4 ODA 监督目标

进攻与防守分类损失为：

$$
\mathcal{L}_{\mathrm{off}}
=
\mathrm{CE}
\!\left(y_t^{\mathrm{off}},p_t^{\mathrm{off}}\right),
$$

$$
\mathcal{L}_{\mathrm{def}}
=
\mathrm{CE}
\!\left(y_t^{\mathrm{def}},p_t^{\mathrm{def}}\right).
$$

ODA 总损失为：

$$
\mathcal{L}_{\mathrm{ODA}}
=
\lambda_{\mathrm{off}}\mathcal{L}_{\mathrm{off}}
+
\lambda_{\mathrm{def}}\mathcal{L}_{\mathrm{def}}.
$$

验证集报告 top-1 准确率、交叉熵、校准误差和注意力熵。公共观察采用与字段物理范围对应的固定特征缩放，模型检查点同时保存缩放方式和标签窗口参数。

## 5. 结果势能模型

### 5.1 时序编码

ODA 编码器为每个观察产生攻防关系表示：

$$
e_t^{\mathrm{ODA}}
=
f_{\theta_{\mathrm{ODA}}}\!\left(o_t\right).
$$

时序编码器聚合最近 $L$ 步关系表示：

$$
h_t^{\mathrm{temporal}}
=
f_{\theta_{\mathrm{temporal}}}
\!\left(
e_{t-L+1}^{\mathrm{ODA}},\ldots,e_t^{\mathrm{ODA}}
\right).
$$

时序编码器可以采用 GRU、LSTM 或因果 Transformer。首个实现采用 GRU，以较小参数规模表达短期比赛记忆。

### 5.2 胜平负分布

结果价值头输出胜、平、负三个 logits：

$$
\ell_t^{\mathrm{WDL}}
=
W_{\mathrm{WDL}}h_t^{\mathrm{temporal}}+b_{\mathrm{WDL}}.
$$

概率分布为：

$$
p_t^{\mathrm{WDL}}
=
\mathrm{softmax}\!\left(\ell_t^{\mathrm{WDL}}\right).
$$

状态势能定义为预期终场效用：

$$
\Phi_{\theta}(x_t)
=
p_t^{\mathrm{win}}-p_t^{\mathrm{loss}}.
$$

平局效用取 0，胜利效用取 1，失败效用取 -1。该定义使势能范围保持在闭区间：

$$
\Phi_{\theta}(x_t)\in[-1,1].
$$

结果分类损失为：

$$
\mathcal{L}_{\mathrm{WDL}}
=
\mathrm{CE}\!\left(y_T,p_t^{\mathrm{WDL}}\right).
$$

一场比赛内的全部运行状态共享该场比赛的终场结果标签。时间与比分进入模型输入，使模型能够学习相同空间局面在各个比赛阶段的结果价值。

### 5.3 左右对称约束

镜像状态的胜负分布满足：

$$
p^{\mathrm{win}}\!\left(\mathcal{M}(x_t)\right)
=
p^{\mathrm{loss}}(x_t),
$$

$$
p^{\mathrm{draw}}\!\left(\mathcal{M}(x_t)\right)
=
p^{\mathrm{draw}}(x_t).
$$

势能满足反对称关系：

$$
\Phi_{\theta}\!\left(\mathcal{M}(x_t)\right)
=
-\Phi_{\theta}(x_t).
$$

对称损失定义为：

$$
\mathcal{L}_{\mathrm{sym}}
=
\left(
\Phi_{\theta}\!\left(\mathcal{M}(x_t)\right)
+
\Phi_{\theta}(x_t)
\right)^2.
$$

势能模型的联合损失为：

$$
\mathcal{L}_{\Phi}
=
\mathcal{L}_{\mathrm{WDL}}
+
\lambda_{\mathrm{sym}}\mathcal{L}_{\mathrm{sym}}
+
\lambda_{\mathrm{ODA}}\mathcal{L}_{\mathrm{ODA}}.
$$

ODA 辅助损失在结果训练期间保持关系表示的攻防语义。结果损失将这些关系映射到真实终场价值。

## 6. 任务奖励与势能奖励

### 6.1 终场任务奖励

终场胜平负定义任务目标。环境奖励为：

$$
r_t^{\mathrm{task}}
=
\begin{cases}
+1, & t+1=T \ \land\ g_T^{\mathrm{own}}>g_T^{\mathrm{opp}}, \\
0,  & t+1<T, \\
0,  & t+1=T \ \land\ g_T^{\mathrm{own}}=g_T^{\mathrm{opp}}, \\
-1, & t+1=T \ \land\ g_T^{\mathrm{own}}<g_T^{\mathrm{opp}}.
\end{cases}
$$

有限时长比赛采用等权时间回报：

$$
\gamma=1.
$$

比赛时间仍然属于观察。策略据此学习比分、剩余时间和场上风险之间的关系。

### 6.2 冻结势能快照

第 $k$ 个 PPO 训练区间使用固定势能快照：

$$
\overline{\Phi}_k(x)
=
\Phi_{\overline{\theta}_k}(x).
$$

缩放势能定义为：

$$
\Psi_k(x)=\beta\overline{\Phi}_k(x).
$$

终止吸收状态的势能定义为：

$$
\Psi_k(x_T)=0.
$$

逐步势能奖励为：

$$
F_t^{(k)}
=
\Psi_k(x_{t+1})-\Psi_k(x_t).
$$

PPO 使用的总训练奖励为：

$$
r_t^{\mathrm{train}}
=
r_t^{\mathrm{task}}+F_t^{(k)}.
$$

### 6.3 望远镜求和性质

一场比赛的势能奖励总和为：

$$
\sum_{t=0}^{T-1}F_t^{(k)}
=
\sum_{t=0}^{T-1}
\left(
\Psi_k(x_{t+1})-\Psi_k(x_t)
\right).
$$

中间状态的势能项逐项抵消，因此：

$$
\sum_{t=0}^{T-1}F_t^{(k)}
=
\Psi_k(x_T)-\Psi_k(x_0).
$$

结合终止势能定义可得：

$$
\sum_{t=0}^{T-1}F_t^{(k)}
=
-\Psi_k(x_0).
$$

对于给定初始状态，势能奖励总和由初始势能唯一确定。策略通过改变终场胜平负提高任务回报，逐步势能奖励负责向前传播局面价值变化。

## 7. Jackaroo 策略网络

### 7.1 实体关系编码

策略网络采用独立参数的实体编码器和攻防注意力层。ODA 预训练参数为策略编码器提供初始化。策略训练随后根据 PPO 目标更新自身参数。

策略时序状态为：

$$
h_t^{\pi}
=
f_{\theta_{\pi,\mathrm{temporal}}}
\!\left(o_{t-L+1},\ldots,o_t\right).
$$

### 7.2 Actor

Actor 输出 32 个原子动作的 logits：

$$
\ell_t^{\pi}
=
W_{\pi}h_t^{\pi}+b_{\pi}.
$$

动作分布为：

$$
\pi_{\theta}(a_t\mid x_t)
=
\mathrm{softmax}\!\left(\ell_t^{\pi}\right).
$$

采样动作提交给引擎指定球员。其余球员继续执行环境提供的球队与个人控制逻辑。

### 7.3 Critic

Critic 估计当前策略在训练奖励下的回报：

$$
V_{\omega}(x_t)
=
W_V h_t^{\pi}+b_V.
$$

冻结势能快照与 PPO Critic 各自承担明确职责。势能模型提供结果导向塑形信号，Critic 跟踪当前策略的训练回报。

## 8. PPO 更新

时间差分残差为：

$$
\delta_t
=
r_t^{\mathrm{train}}
+
V_{\omega}(x_{t+1})
-
V_{\omega}(x_t).
$$

GAE 优势估计为：

$$
\widehat{A}_t
=
\sum_{l=0}^{T-t-1}
\lambda^l\delta_{t+l}.
$$

策略概率比为：

$$
\rho_t(\theta)
=
\frac{
\pi_{\theta}(a_t\mid x_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid x_t)
}.
$$

PPO clipped objective 为：

$$
\mathcal{J}_{\mathrm{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\widehat{A}_t,
\mathrm{clip}
\left(
\rho_t(\theta),1-\epsilon,1+\epsilon
\right)
\widehat{A}_t
\right)
\right].
$$

Critic 损失为：

$$
\mathcal{L}_V(\omega)
=
\mathbb{E}_t
\left[
\left(
V_{\omega}(x_t)-\widehat{R}_t
\right)^2
\right].
$$

策略熵为：

$$
\mathcal{H}(\pi_{\theta})
=
-\mathbb{E}_t
\left[
\sum_a
\pi_{\theta}(a\mid x_t)
\log\pi_{\theta}(a\mid x_t)
\right].
$$

PPO 优化目标组合策略、价值与熵项：

$$
\mathcal{L}_{\mathrm{PPO}}
=
-\mathcal{J}_{\mathrm{clip}}
+
c_V\mathcal{L}_V
-
c_H\mathcal{H}(\pi_{\theta}).
$$

## 9. 交替训练流程

### 9.1 初始化阶段

1. 运行内置 AI 对局并生成完整轨迹。
2. 从轨迹构造攻防注意力标签。
3. 训练 ODA 编码器与攻防分类头。
4. 使用完整比赛的胜平负标签训练结果势能模型。
5. 将 ODA 参数复制到 Jackaroo 策略编码器作为初始化。
6. 创建首个冻结势能快照。

### 9.2 策略阶段

1. Jackaroo 使用当前策略运行完整比赛。
2. 训练器通过冻结势能快照计算每个决策步的势能差。
3. 训练器将终场任务奖励与势能差相加。
4. PPO 使用收集到的轨迹更新 Actor 与 Critic。
5. 完整比赛轨迹进入结果数据集。

### 9.3 势能刷新阶段

1. 结果数据集混合初始内置 AI 轨迹与近期 Jackaroo 轨迹。
2. 结果势能模型使用胜平负标签继续训练。
3. ODA 辅助损失维持攻防关系表示。
4. 验证集的结果交叉熵与平均势能变化决定是否接受候选模型。
5. 接受的参数经过插值后形成下一个 PPO 训练区间的冻结势能快照。

交替流程写为：

$$
\mathrm{Collect}
\rightarrow
\mathrm{OptimizePPO}
\rightarrow
\mathrm{FitPotential}
\rightarrow
\mathrm{FreezePotential}
\rightarrow
\mathrm{Collect}.
$$

每个 PPO 区间内的势能函数保持固定。势能刷新发生在完整区间边界。

## 10. 数据划分与采样

数据集按比赛划分为训练集、验证集和测试集。同一场比赛的全部决策步属于同一个划分，以维持统计独立性。划分由比赛结果内的确定性计数决定，因此续训前后保持一致。

结果数据集采用分层采样，使胜、平、负样本在每个批次中保持稳定比例。时间分层采样同时覆盖开局、中场和终场阶段。比分分层采样覆盖落后、持平和领先状态。

初始内置 AI 轨迹提供基础局面，近期 Jackaroo 轨迹覆盖当前策略的状态分布。固定容量的近期轨迹集合控制训练成本，初始轨迹与近期轨迹共同参与按比赛划分的数据集。

## 11. 评估指标

### 11.1 ODA 指标

- 进攻目标 top-1 与 top-3 准确率。
- 防守目标 top-1 与 top-3 准确率。
- 交叉熵。
- 期望校准误差。
- 注意力熵。

### 11.2 势能指标

- 胜平负准确率。
- 多分类 Brier score。
- 结果交叉熵。
- 镜像对称误差。
- 按比赛时间与比分分组的校准曲线。

Brier score 定义为：

$$
\mathrm{Brier}
=
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{c\in\{\mathrm{win},\mathrm{draw},\mathrm{loss}\}}
\left(
p_{n,c}-y_{n,c}
\right)^2.
$$

### 11.3 策略指标

- 给定种子序列上的胜率、平率、负率和平均净胜球。
- 各 PPO 更新区间的策略损失、价值损失和策略熵。
- 终场任务回报。
- 每场势能奖励总和与理论常数的数值误差。
- 动作分布、控球转换和射门事件统计。

正式策略评价采用终场任务回报。势能奖励与 ODA 指标用于训练诊断。

## 12. 消融实验

三个训练组采用相同网络容量、PPO 更新设置、随机种子和优化超参数：

1. Terminal PPO 使用终场胜平负奖励。
2. Learned Potential PPO 使用结果监督势能与通用实体编码器。
3. ODA-Conditioned Learned Potential PPO 使用 ODA 表示、结果监督势能和势能差奖励。

第一组衡量稀疏结果奖励的基础表现。第二组衡量 learned potential 的贡献。第三组衡量 ODA 表示对势能质量和策略学习的贡献。

## 13. 潜在问题与解决方案

### 13.1 势能分布偏移

势能模型从内置 AI 轨迹开始学习，Jackaroo 进入较少访问的局面后会产生分布偏移。训练数据混合初始内置 AI 轨迹与固定容量的近期 Jackaroo 轨迹。初始集合维持基础局面，近期集合覆盖当前策略的状态分布，按比赛划分的验证集衡量结果预测质量。

势能模型使用 $K$ 个独立初始化的成员估计预测分歧：

$$
\bar{\Phi}(x)
=
\frac{1}{K}
\sum_{j=1}^{K}
\Phi_j(x),
\qquad
\sigma_{\Phi}^{2}(x)
=
\frac{1}{K}
\sum_{j=1}^{K}
\left(\Phi_j(x)-\bar{\Phi}(x)\right)^2.
$$

有效势能采用不确定性置信系数：

$$
\Psi(x)
=
\beta c(x)\bar{\Phi}(x),
\qquad
c(x)
=
\exp\left(-\lambda\sigma_{\Phi}^{2}(x)\right).
$$

成员分歧较大的局面自动获得较小的 shaping 权重。正式评价继续使用比赛终场结果，势能指标承担信用分配与诊断职责。

### 13.2 ODA 标签质量

攻防注意力标签由公共观察序列生成。标签生成器综合球队球权、最近触球者、球员到球时间和连续观测，在前向窗口内寻找连续稳定持球的目标球员。争抢、解围和球权归属模糊的时间段获得较低标签置信度。

ODA 分类使用置信度加权交叉熵：

$$
\mathcal{L}_{\mathrm{ODA}}
=
-\sum_t w_t
\sum_i y_{t,i}\log p_{t,i},
\qquad
w_t\in[0,1].
$$

标签与置信权重在势能训练批次中即时计算。验证集上的攻防 top-1、top-3、交叉熵、校准误差与注意力熵共同描述标签和表示质量。

### 13.3 势能刷新与 PPO 稳定性

每个 PPO 区间使用一份冻结势能快照。采样、奖励计算和 PPO 更新共享同一份 $\Psi_k$，候选势能在区间边界完成训练与验证后进入下一轮。

势能更新顺序为：

1. 冻结 $\Psi_k$。
2. 使用 $\Psi_k$ 收集完整 rollout。
3. 使用 $\Psi_k$ 计算 rollout 中每一步的 shaping reward。
4. 使用该 rollout 完成 PPO 更新。
5. 训练候选势能并在验证集上计算结果交叉熵与平均势能变化。
6. 生成下一个区间的 $\Psi_{k+1}$。

区间之间可以使用插值限制变化幅度：

$$
\Psi_{k+1}
=
(1-\alpha)\Psi_k
+
\alpha\widehat{\Psi}_{k+1},
\qquad
\alpha\in[0,1].
$$

验证集上的平均变化量作为刷新门槛：

$$
\frac{1}{|\mathcal{V}|}
\sum_{x\in\mathcal{V}}
\left|
\Psi_{k+1}(x)-\Psi_k(x)
\right|
\leq\epsilon.
$$

### 13.4 势能差的累计性质

势能差采用与 PPO 相同的折扣因子：

$$
F_t
=
\gamma\Psi(x_{t+1})-\Psi(x_t).
$$

其折扣累计值满足：

$$
\sum_{t=0}^{T-1}\gamma^tF_t
=
-\Psi(x_0)
+
\gamma^T\Psi(x_T).
$$

终止状态势能设为零时，累计 shaping 项归结为初始势能项。随机初始状态产生的差异由 Critic 的状态基线和标准 advantage 归一化吸收。$\beta$ 控制单步 shaping 的优化尺度。

### 13.5 左右侧对称性

训练环境按比赛交替分配物理左右侧。受控侧为右侧时，原生环境输出经过 180° 旋转，并将受控队伍放入观察的第 0 队列；方向动作按照同一旋转映射提交给物理引擎。策略始终使用“己方进攻方向为正 $x$ 轴”的规范坐标。

镜像变换 $\mathcal{M}$ 用于检验势能的结果对称性：

$$
\Phi_{\theta}(\mathcal{M}(x))
=
-\Phi_{\theta}(x).
$$

训练中采用软对称损失：

$$
\mathcal{L}_{\mathrm{sym}}
=
\left(
\Phi_{\theta}(\mathcal{M}(x))
+
\Phi_{\theta}(x)
\right)^2.
$$

总损失写为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{outcome}}
+
\lambda_{\mathrm{sym}}\mathcal{L}_{\mathrm{sym}}.
$$

### 13.6 动作空间与观察历史

Jackaroo 使用引擎的 32 个原子动作。32 个动作始终构成策略分布，当前比赛状态决定动作产生的物理效果。观察中的 `action_frame`、`touch_frame`、粘滞动作与球员速度提供动作延续所需的上下文。观察编码保留完整公共字段，历史建模采用固定窗口 $L$，窗口长度由训练参数 `--history-length` 指定。

### 13.7 训练验证与漏洞监控

势能模型提供信用分配信号，终场胜平负提供最终任务指标。训练记录终场结果、势能预测指标、势能累计误差、动作分布、控球转换和射门事件。候选势能的结果交叉熵或平均变化超过门槛时，训练器恢复上一份势能参数并降低 $\beta$。Terminal PPO、Learned Potential PPO 与 ODA-Conditioned Learned Potential PPO 通过相同训练参数和评价种子完成消融比较。

## 14. 实现结构

- `dataset.py` 负责完整轨迹存储、镜像样本、标签构造和数据划分。
- `attention.py` 负责实体编码、对手注意力、队友注意力和 ODA 分类头。
- `potential.py` 负责时序编码、胜平负头、势能计算和势能快照。
- `network.py` 负责 Jackaroo Actor-Critic。
- `env.py` 负责公共观察、终场任务奖励和势能差组合。
- `ppo.py` 负责 rollout、GAE 和 PPO 更新。
- `train.py` 负责交替训练调度与检查点管理。
- `evaluate.py` 负责给定种子序列上的正式评价。

模型接口维持现有 32 原子动作和完整公共观察。训练模块通过字段复制构造张量，势能模型与策略模型维护独立参数。

## 15. 参考资料

- [Offensive and Defensive Attention Reward in Reinforcement Learning for Football Game](./ODAR.pdf)
- [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](./PBRS.pdf)
- [Reinforcement Learning with Local Shaping Rewards](./DIS.pdf)
