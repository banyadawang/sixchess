# AlphaGO_Connect6_end — 开发文档

> 版本：v2（2026-10-21 更新）　|　适用于仓库当前代码状态
> 本文档面向开发者与维护者，覆盖项目全貌与各模块职责；具体算法细节请直接阅读对应源码注释。

---

## 1. 项目概述

核心思路：

- **动作空间自回归分解**：一回合两子不是简单的 361² 联合空间，而是分解为 `P(a1,a2|s) = P(a1|s) · P(a2|s,a1)`——先预测第一子，再以第一子为条件预测第二子。
- **搜索 + 神经网络**：C++/OpenMP 实现的 MCTS（PUCT）在 CPU 上并行搜索，每批叶子由 TensorRT 引擎在 GPU 上推理给出策略与价值。
- **训练闭环**：监督预训练 → 自对弈生成 → 强化学习训练 → 真实搜索预算下的配对换色门禁 → 晋升为新代模型，循环迭代。

模型：**C6TransNet**（约 11M 参数），17 输入平面，embed 256，6 层带 2D 相对位置编码的注意力 Transformer + 空洞卷积 stem，三个输出头：第一子策略、条件第二子策略、价值。

仓库不包含权重/引擎/棋谱数据（均 gitignore）：模型权重需从 ModelScope 下载或按文档在本机训练生成，TensorRT 引擎需本机构建。

---

## 2. 总体架构与数据流

```
┌──────────────┐   初始权重    ┌─────────────────────────────────────────┐
│ supervised_   │ ──────────▶ │           reinforcement_learning          │
│ learning      │ checkpoint  │  run_loop.py / run_forever.py 逐代循环:    │
│ (监督预训练)   │             │  构建引擎 → generate 自对弈 → 合并缓冲     │
└──────────────┘             │  → train 训练 → evaluate 门禁 → 晋升        │
                             └───────┬─────────────────────┬─────────────┘
                                     │ best.pth             │ current_model.engine
                          ┌──────────▼──────┐      ┌───────▼────────┐
                          │  local/ (GUI)   │      │ Competition/    │
                          │  PyQt5 桌面客户端 │      │ (Web 对弈/演示) │
                          └─────────────────┘      └────────────────┘
                                     │                       │
                          ┌──────────▼───────────────────────▼──────────┐
                          │   C++ MCTS (mcts_engine.cpp) + ctypes 封装    │
                          │   TensorRT 推理 (MCTSEngine)                  │
                          └─────────────────────────────────────────────┘
```

两条主线：

1. **训练线**（Linux/WSL2 + NVIDIA GPU）：`supervised_learning/train.py` → `reinforcement_learning/run_loop.py`。每代自对弈棋谱以 CSV（moves / winner / policies / bonuses）落盘，合并进 replay 缓冲后训练候选模型，通过门禁才替换 `best.pth` 与 `current_model.engine`。
2. **推理线**：Python（PyQt5 GUI 或 Flask Web）持有权威棋盘状态，通过 ctypes 调用 C++ MCTS 库；MCTS 每展开一批叶子就把棋盘编码交给 TensorRT 引擎推理，取回策略/价值继续搜索。

---

## 3. 目录结构

| 目录 | 职责 |
|------|------|
| `supervised_learning/` | 监督预训练：人类棋谱 → C6TransNet 初始权重 |
| `reinforcement_learning/` | **生产主模块**：模型、MCTS、TensorRT 封装、自对弈与训练流水线、门禁 |
| `experiments/` | 与生产隔离的架构实验（`nebula_v3`、`pair_policy`） |
| `benchmarks/` | 性能基准：MCTS 吞吐、TRT batch 曲线、训练吞吐等 |
| `tests/` + `scripts_smoke_env.py` | 单元测试与环境冒烟脚本 |
| `Competition/` | Flask Web 对弈与实时搜索观察台（比赛演示） |
| `local/` | Windows 桌面客户端（PyQt5 GUI + 本地 MCTS 编译产物） |
| `final/` | 历史各代模型的循环赛（擂台制），产出历代棋力排序 |

---

## 4. 环境搭建

### 4.1 依赖

- Python 3.10+（推荐 3.10，requirements 在其上验证）
- NVIDIA GPU + 驱动（CUDA 12.8 对应驱动 570+）
- `pip install -r requirements.txt`：torch 2.11.0+cu128、tensorrt-cu12==11.1.0.106、onnxruntime-gpu、PyQt5、Flask 等
- C++ 编译器 + OpenMP：Linux 用 g++，Windows 用 MSVC（VS 2022 Build Tools）

### 4.2 编译 MCTS 动态库

- 主仓库（训练/Web 用）：`python reinforcement_learning/core/compile_mcts.py` → 生成 `libmcts.so`
- 本地 GUI（Windows 用）：`python local/mcts/compile_dll.py` → 生成 `local/core/mcts.dll`。脚本会自动通过 vswhere 定位 VS 2022 并加载编译环境，普通终端即可运行。

### 4.3 模型权重与引擎

- 权重：ModelScope 下载或本机训练；`reinforcement_learning/config.py` 会自动优先捡 `supervised_learning/checkpoints/checkpoint_latest.pth` 作为初始模型。
- 引擎：`pipeline/export_onnx.py <pth> <onnx>` 导出 ONNX，`pipeline/build_engine.py <onnx> <engine>` 构建 TensorRT 引擎。
- **重要**：TensorRT 引擎与 GPU 型号、TensorRT 版本强绑定，换机/换 TRT 版本必须重建。TRT 11.x 已移除 FP16 BuilderFlag（默认 TF32/混合精度）；构建时优化 profile 的 max batch 建议 ≤256（app 实际 batch=32）。

---

## 5. 模块详解

### 5.1 supervised_learning/ — 监督预训练

训练链第一环。在人类对局棋谱上预训练 C6TransNet，为 RL 提供初始权重。

- 入口：`python train.py --data data/processed/connect6_cleaned.csv --epochs 50`
- 损失 = 双头策略交叉熵（teacher forcing）+ 0.5×value MSE；AdamW + 余弦退火 + AMP。
- 产出：`checkpoints/checkpoint_latest.pth`（RL 初始模型）。
- 关键文件：`train.py`、`models/c6transnet.py`（模型）、`data/dataset.py`（17 平面特征）、`data/connect6_game.py`（规则）。

### 5.2 reinforcement_learning/ — 生产主模块

项目核心，分两层：

**core/**（引擎与算法）：
- `model.py`：`C6TransNet` 生产模型（含 `RelativeGlobalAttention` 2D 相对位置注意力等组件）
- `mcts.py`：`MCTSEngine`（ctypes 封装 C++ MCTS + TensorRT 推理、CUDA Graph、批量局面去重）、`MCTSGameContext`（单 worker 多盘并发合批）
- `mcts_engine.cpp` + `c6_logic.h`：C++17/OpenMP MCTS 核心（PUCT、渐进展开、PV/胜率统计）
- `connect6_game.py`：六子棋规则/棋盘状态
- `gpu_vcdt.py`、`trt_relative_attention.py`：GPU 威胁检测、Triton 自定义 TRT 插件（实验性，默认关）

**pipeline/**（各阶段独立脚本）：
- `generate.py`：自对弈产棋谱（只产数据不改模型）
- `train.py`：候选模型训练（bf16/fp16 AMP）
- `evaluate.py`：门禁对战（配对换色、固定开局套件）
- `export_onnx.py` / `build_engine.py`：模型 → TRT 引擎
- `remote_generate.py`：远端机器自对弈执行器
- `analyze_data.py`：棋谱/缓冲统计

**逐代循环**：
- `run_loop.py`：单代流程编排（构建 → 自对弈 → 合并 → 训练 → 门禁 → 晋升）
- `run_forever.py`：常驻监督器，每代重启子进程刷新配置与 SwanLab 认证

**门禁标准**：黑方 400 / 白方 1200 搜索预算，`EVAL_SEED=2026` 固定开局套件，胜率 ≥50% 且白方胜率 ≥10% 才晋升；并设有 model_gen_10/100/200 静态基准与代差偏移 [10,20,50] 的参考对局。

**配置**：`config.py` 集中管理，约 60 个 `NEBULA_*` 环境变量可覆盖（详见第 7 节）。

### 5.3 experiments/ — 架构实验（与生产隔离）

- `nebula_v3/`：V3/V4 候选架构（ConvNeXt 大核、2D RoPE、SwiGLU、FlashAttention、策略蒸馏）。结论：未通过门禁，生产仍用 V2。
- `pair_policy/`：原生双落子策略实验——一次主干推理同时输出第一子策略、价值与第二子条件策略因子（rank-16），白方高价值节点再刷新完整网络。吞吐较完整网络高约 46%，但多轮门禁未通过，未替换主线。
- 二者均有独立 `run_loop.py`/`run_forever.py`，不自动覆盖生产 checkpoint。

### 5.4 benchmarks/ — 性能基准

固定搜索预算/固定模型下对比吞吐与棋力，防止单算子优化与端到端表现脱节。原则：单算子更快必须回跑端到端 MCTS 验证。

- `benchmark_mcts.py`：端到端 MCTS 吞吐（spawn 多进程、每进程独立 TRT context/CUDA Graph）
- `compare_mcts_configs.py`：两组 MCTS 参数配对换色对局比棋力
- `benchmark_tensorrt_engine.py`：不同 batch 的推理吞吐/显存曲线
- `benchmark_training.py`：BF16 双卡训练吞吐（随机数据）
- 其余：多局合批、Triton 相对位置注意力单算子、pair 头 int8 AVX2 微基准

### 5.5 tests/ 与 scripts_smoke_env.py — 质量门禁

- 模型一致性（融合 vs 显式注意力）、C++ MCTS ABI 与 eval 缓存（现场 g++ 编译临时 .so）、数据集/平衡控制器/pair 实验逻辑、RL 远端自对弈 mock 测试。
- 冒烟脚本 5 步检查：环境（torch/CUDA/TRT/libmcts）→ 规则 → 模型前向 → ctypes 加载 → 核心单测。
- 大部分测试不需要 GPU。

### 5.6 Competition/ — Web 对弈与演示台

- 入口：`cd Competition/web && python app.py` → http://127.0.0.1:5000
- Flask 后端（`app.py`）维护权威棋盘，按 ~256 模拟/块分片驱动 C++ MCTS，REST 轮询实时返回搜索进度/胜率/候选点；前端为原生 HTML/Canvas（`templates/index.html`），自适应桌面与移动端。
- `ai/agent.py`：可中断思考的 AI 代理（限时或固定模拟、后台 pondering）。
- `utils/recorder.py`：比赛格式棋谱输出。
- 历史实现（已废弃，勿再使用）：`web/mcts_core_patch.py`、`core/mcts_engine.cpp`、`core/c6_logic.h`、`templates/index_v1.html`——旧版 17 平面独立推理包装器，已被主线"融合原始棋盘输入"引擎取代。
- 环境变量：`NEBULA_WEB_CHECKPOINT_DIR`、`NEBULA_WEB_DEFAULT_ENGINE`、`NEBULA_WEB_HOST`、`NEBULA_WEB_PORT`。

### 5.7 local/ — Windows 桌面客户端（GUI）

- 入口：`cd local && python main.py`
- 结构：`main.py`（入口与运行时兜底）→ `ui/`（`main_window.py` 主窗口、`game_board.py` 棋盘控件、`dialogs.py` 结果对话框）→ `core/`（`ai_interface.py` AI 工作线程、`ai_mcts.py` MCTS/TRT 封装、`mcts_strategy.py` 搜索策略、`opening_book.py` 开局库、`sgf_handler.py` 棋谱）。
- 功能：普通模式（人机对战）与团队轮换模式（AI 协助己方轮流落子）、动态思考/深度思考/后台思考（多实例 Ponder + 树复用）、开局库、策略热力图、调试暂停（AI 落子前人工确认）、SGF 棋谱保存/载入、加时。
- `mcts/`：GUI 专用的 C++ MCTS 源码与编译脚本（编译产物 `core/mcts.dll`）。
- `tools/`：模型/引擎构建工具链（`model.py`、`export_onnx.py`、`build_engine.py`、`best.pth` 权重）。
- **v2 改版（2026-10-21）**：界面切换为"深空主题"（深空蓝紫配色 + 暗夜石板蓝棋盘 + 金色落子标记），窗口标题与面板底部标注 "AlphaGO_Connect6_end v2"；功能与布局与 v1 一致。
- Windows 专项说明见第 8 节常见问题。

### 5.8 final/ — 历史模型循环赛

- 入口：`python final/tournament_pro.py`
- 对 `reinforcement_learning/checkpoints/model_gen_*.pth` 最近 100 个代际模型做擂台制循环赛（Init → Qualifiers → Semis → Elite8 → Finals），黑白互换双循环。
- 产出：`tournament_state.json`（断点可续）、`logs/*.csv` 棋谱、`results/summary_*.txt` 积分榜。

---

## 6. 关键技术与约定

- **搜索**：PUCT MCTS + 渐进展开 + 剪枝 K 值动态调整 + 动态提前熔断（默认关）+ 对手回合后台思考（Ponder，多实例树复用）。
- **推理**：TensorRT 引擎（TRT 11.1，TF32/混合精度），批量叶子一次推理；CUDA Graph 与页锁定缓冲降低启动开销；自对弈支持融合图（`TRT_FUSED_SELFPLAY`，默认关）。
- **训练**：AdamW + 余弦退火 + AMP（bf16）；SwanLab 双层实验可视化（逐代 gen_N + 总览 AlphaZero_Training_Loop）。
- **工程约定**：权重/引擎/棋谱/机器专用脚本均不入库；门禁是模型晋升唯一通道；改 `MCTS_BATCH_SIZE` 后必须重建引擎；引擎与 GPU/TRT 版本绑定。

---

## 7. 常用环境变量

| 变量 | 含义 |
|------|------|
| `NEBULA_SELFPLAY_GPUS` | 自对弈可见 GPU，如 `0,1` |
| `NEBULA_NUM_WORKERS` | worker 数（通常每卡 1 个） |
| `NEBULA_MCTS_THREADS` | 每 worker 的 MCTS CPU 线程 |
| `NEBULA_MCTS_BATCH_SIZE` | 推理 batch（改后需重建引擎） |
| `NEBULA_SIMULATIONS_BLACK/WHITE` | 黑白方搜索预算 |
| `NEBULA_TRT_FUSED_SELFPLAY` | 自对弈融合推理图开关 |
| `NEBULA_CUDA_GRAPH` | CUDA Graph 开关 |
| `NEBULA_SELFPLAY_BACKEND` | 自对弈后端（local/remote） |
| `NEBULA_REMOTE_*` | 远端自对弈（HOST/PROJECT_DIR/PYTHON/GPUS/NUM_WORKERS 等） |
| `NEBULA_TRAINING_GPU(S)` | 训练卡 |
| `NEBULA_TRAIN_PRECISION` | 训练精度 bf16/fp16/fp32 |
| `NEBULA_SWANLAB_MODE` | online/offline/local/disabled |
| `NEBULA_SEED` / `EVAL_SEED` | 随机种子 / 门禁开局套件种子 |
| `NEBULA_PAIR_*` | pair_policy 实验专用（HEADS、REFRESH_VISITS 等） |
| `NEBULA_WEB_*` | Competition Web 服务（CHECKPOINT_DIR、DEFAULT_ENGINE、HOST、PORT） |

完整列表与默认值见 `reinforcement_learning/config.py`。
