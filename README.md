# 思维点和分歧度是模型自我持续学习的依据 (DATP & DAM)

[![GitHub License](https://img.shields.io/github/license/QZL2004qz1/DATP_DAM)](LICENSE)


针对深度学习模型在部署后**无法实时持续自我学习**（面临灾难性遗忘与高昂微调成本）以及 Transformer **自注意力机制计算复杂度随序列长度平方级膨胀 ($O(n^2)$)** 的双重挑战，本文受到人类认知科学中**双系统理论（Dual-Process Theory）**的启发，提出了全新的学术与工程框架：**分聚思维点网络（DATP）** 与 **分歧聚合机制（DAM）**。


---

## 🚀 核心亮点

1. **零参数冻结基座持续学习 (DATP)**
   - 受到大脑双系统理论（快思考与慢思考）的启发。将预训练完成的通用基座模型参数完全冻结（系统 1），通过解耦的**审判官网络**感知认知边界，在非参数空间中引入动态演化的**思维点池（Thought-Point Pool）**（系统 2）。
   - 赋予模型在**不修改基座参数、不触发灾难性遗忘**前提下的流式即时纠错与持续自我学习能力。
2. **全新注意力机制替代算子 (DAM)**
   - 彻底摒弃传统的 Q、K 矩阵乘法，以**分歧度（Divergence）**替代传统的相似度度量。
   - 成功将自回归推理的时间复杂度降至 **$O(nd)$**，并将推理显存占用从标准注意力的 $O(nd)$ 降至与序列长度完全无关的 **$O(d)$ 恒定水平**（仅需维护两个 $d$ 维累积向量），在边缘设备及长文本场景中优势显著。
3. **生物启发式演化动力学**
   - 引入动态**空间聚合**与**生物遗忘**机制。将散落的“孤立错题点”逐步归纳凝聚为局部知识区域，并自动修剪掉由于噪声引起的低健康度思维点，保证思维点池的有界性与泛化能力。

---

## 📐 架构设计

### 1. 思维点与分歧度认知闭环
传统神经网络的“学习 $\rightarrow$ 预测”是单向过程，而 DATP 实现了“预测 $\rightarrow$ 评价 $\rightarrow$ 学习 $\rightarrow$ 再预测”的循环过程，使知识在交互中正向积累。

### 2. DATP 整体路由决策
模型通过解耦审判官评估基座模型的犯错概率 $P(\text{Wrong}|x)$,从而实现按需计算（Compute-on-Demand）：
- **快速通道（系统 1）**：审判官判定低风险，直接由冻结基座网络输出，延迟控制在极低水平。
- **慢速通道（系统 2）**：审判官判定高风险，计算分歧度。若命中已有思维点，则调用局部专家进行**残差补丁修正**；若未命中，则即时创建新思维点进行**在线即时学习**。

#### 🖼️ DATP 架构设计图
<p align="center">
  <img src="assets/datp_architecture.png" alt="DATP Architecture" width="85%">
</p>

---

### 3. DAM 算子与传统 SDPA 机制对比
DAM 算子通过引入分歧度聚合，将标准自注意力机制（SDPA）的平方级复杂度打破，实现更高效的流式状态更新。

#### 🖼️ 架构机制对比
<table>
  <tr>
    <td align="center"><b>标准自注意力机制 (SDPA)</b></td>
    <td align="center"><b>分歧聚合机制 (DAM)</b></td>
  </tr>
  <tr>
    <td><img src="assets/sdpa_architecture.png" alt="SDPA Architecture" width="100%"></td>
    <td><img src="assets/dam_architecture.png" alt="DAM Architecture" width="100%"></td>
  </tr>
</table>

---

## 📈 实验结果

- **语言建模 (WikiText-103)**：在 84MB 参数量、1024 序列长度条件下，DAM 算子达到了 **20.69** 的困惑度（Perplexity），保持了与标准注意力机制可比的能力，同时实现线性的推理显存开销。
- **回归任务**：在五个标准回归数据集上，DATP 演化动力学成功将思维点数量压缩约 **72% ~ 75%**，单样本推理延迟控制在 **0.08 - 0.63 毫秒**，同时预测性能保持稳定。

---

## 🛠️ 快速开始

### 1. 环境安装
```bash
git clone https://github.com/QZL2004qzl/DATP_DAM.git
cd DATP_DAM
pip install -r requirements.txt

# 复现DATP算法结果
python DATP.py

# 复现DAM算法结果
python DAM.PY
