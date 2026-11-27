# re-tokendial · Wan 训练、生成与评分

## 最新实验 · 单视频训练「人物变老」

使用 Wan2.1-T2V-1.3B 和 TokenDial 的人物变老方向。固定初始化种子 `20261123`、AdamW 和学习率 `5e-4`。训练设置为 832×480、17 帧。同一条视频每轮使用两次，共 12 轮、24 次参数更新。

仓库附带本次训练实际使用的 [预处理视频 clip_00001.mp4](experiments/person_older/clip_00001.mp4)。[训练清单](experiments/person_older/metadata.csv) 记录其 caption、来源和 SHA256。克隆仓库后，从仓库根目录运行下列命令。`verify` 会先检查视频哈希，训练直接读取仓库内的文件。请先按下文激活 Conda 环境，并准备 Wan、BERT 和 InternVideo2 权重。

片段来自 [Soul-AILab 的 VividHead 数据集](https://huggingface.co/datasets/Soul-AILab/VividHead)（样本 `VividHead:1`）。这里提供从原素材 0.5 秒处开始裁剪的实验用预处理片段。数据集页面标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，并说明其用途为研究和非商业。

```bash
export BASELINE_BERT=/models/bert-large-uncased
export INTERNVIDEO_CKPT=/models/InternVideo2-stage2_1b-224p-f4.pt
CLIPS=experiments/person_older

python reproduce.py verify person_older --study single --clips "$CLIPS"
python reproduce.py train person_older --study single --clips "$CLIPS"
python reproduce.py generate person_older --study single
python reproduce.py score person_older --study single
```

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | 视频、caption 和外观训练权重 | `training/checkpoints/`、`step_loss.csv`、`loss_curve.json` 和恢复断点 |
| 推理 | 训练 checkpoint 和冻结的 prompt | `unseen6/native/nomask/` 下 18 个 MP4，以及含 SHA256 的 `manifest.json` |
| 评测 | 生成清单、CLIP、AlexNet LPIPS 和人物变老 guidance | `metrics/metrics.csv`、`metrics/summary.json`、`alignment/alignment.json` |

输出根目录为 `runs/person_older/matched/single/`。默认使用第 12 轮的 checkpoint `training/checkpoints/epoch-11.safetensors`，文件名从 0 开始。已有 checkpoint 时可直接用 `generate --checkpoint /path/to/weights.safetensors`，更换权重请使用独立的 `--work` 目录。训练中断后重跑相同命令可使用断点。

[推理协议](experiments/person_older/protocol.json) 固定六条未见 prompt、seed 2、强度 0 / 1 / 3、无 mask、832×480、41 帧、24 fps、50 个采样步和文本 CFG 4.5。复跑原 caption 和咖啡馆的快速验证时，在生成和评分命令中都加 `--prompt-set smoke`。

## 原有 12 属性 Wan 复现实验

本仓库整理了 2026 年 11 月这一轮 TokenDial **公开 Wan 实现**的复现实验。实验从 Table 7 的 12 个概念中各选一个属性，自训视觉 token 偏移权重，再用相同 prompt、seed 和五档强度生成视频。论文未在 Table 2 主实验设置中明确给出骨干权重 ID，主表汇总范围也更广；这里可以复跑我们的 Wan 实验，并与论文报告值作参考对照。

## Table 7 全属性评测（已有 T2V 视频）

Table 7 逐项列出 **12 个概念、57 个属性方向**。本仓库的历史训练实验覆盖其中 12 个方向。全属性入口用于列清单和检查已有视频的覆盖，再调用现有四指标与作者对齐程序。每个新方向的训练数据、checkpoint 和实际注入强度需另行准备并记录。

以下命令从仓库根目录开始，在上述 Conda 环境中执行。清单检查仅需 Python 标准库，可在本地运行。示例 seed 2、3 是复现选择；每个属性使用作者材料中对应的 8 条 prompt，每条两个 seed、五个逻辑档。§6 描述每属性 16 个基础视频，附录 C.2 描述 16 条 prompt 各用两个 seed，这里的 8×2 是明确选定的协议。[论文 §6、Table 7、附录 C.2](https://arxiv.org/pdf/2603.27520v2)

```bash
cd TokenDial
python -m evaluation table7 --seeds 2 3 > table7_inventory.json
```

清单给出每项的概念、属性、manifest 用名、prompt 文件及可用的评分 guidance。仓库自带作者提供的 56 个 JSON。人物步态用 `person_walk.txt`，manifest 写 `concept=person_walk`、`attribute=walk_to_run`，论文概念仍为 person。评分前将 guidance 目录复制到工作目录，补入作者文件或注明来源的自拟版本。自拟时应分别报告原始 56 项子集和含补写项的 57 项结果。

```bash
cp -R evaluation/guidance ./guidance_full
cp /path/to/walk_to_run.json ./guidance_full/person/walk_to_run.json
python -m evaluation table7 --manifest /data/full/manifest.json --guidance-dir guidance_full --seeds 2 3
python -m evaluation evaluate --manifest /data/full/manifest.json --guidance-dir guidance_full --out-dir /data/full/metrics --device cpu
python -m evaluation align --manifest /data/full/manifest.json --guidance-dir guidance_full --metrics-csv /data/full/metrics/metrics.csv --out-dir /data/full/alignment --device cpu
```

按[评测清单格式](TokenDial/evaluation/README.md)准备一个 manifest。method 使用 `nomask`、`mask`；prompt 填对应文件中的整行，seed 填整数。每条轨迹记录逻辑档 0–4 及实际 model_scale，同一属性的两组须使用同一套实际强度。每组应有 912 条轨迹、4,560 个档位记录，零档文件可跨组共享。检查缺项或重复时命令以非零状态退出。评分后的 summary 按 method 汇总；先核对 alignment.json，再分别比较论文 Table 2 的两行。该清单检查本身不会训练、生成、计算分数或验证视觉编辑效果。



## 模块的输入和输出

| 模块 | 输入 | 输出 |
|---|---|---|
| 训练 · `reproduce.py train` | Wan2.1-T2V-1.3B、逐属性视频与字幕清单、外观方向或运动损失配置 | 每轮训练的 `.safetensors`，本轮取 `epoch-1.safetensors` |
| 生成 · `reproduce.py generate` | checkpoint、冻结的 prompt/seed、五档 scale、mask 配置 | 分档 MP4、含路径和 SHA256 的 `manifest.json`、外观 mask 及步序记录 |
| 四指标 · `reproduce.py score` | 生成清单、概念与属性的 guidance、CLIP 和 AlexNet LPIPS 权重 | `metrics/metrics.csv`、`metrics/summary.json`、作者实现一致性检查 `alignment/alignment.json` |
| 原作者 T2V VLM · `TokenDial/evaluation/vendor/gemini_slider_video_eval.py` | 同一 prompt 和 seed 的五档视频，第一档是生成基准 | Gemini 对提示遵循、主体保持、背景一致性、强度递进的分数及理由，CSV 和 JSONL |
| 项目 TV2V VLM · `tv2v_vlm/gemini_tv2v_eval.py` | **独立源视频**和五档编辑输出，按 −4、−2、0、2、4 排列 | 上述四项及其均分、单列的源视频保持分、逐时间点评语和协议记录 |

## 从头复跑一个属性

以下命令在仓库根目录用 Linux Bash 执行。单机入口适用于已分配的 GPU。训练和生成使用 CUDA，四指标使用 CPU。服务器原有的独占卡调度代码保存在 `artifacts/run/`；单机入口可直接调用底层相同的生成、mask 和评分实现。

**1. 创建 Conda 环境并准备输入。** 以下命令适用于 Linux。使用与驱动兼容的 PyTorch CUDA 构建，H20 验证使用 Python 3.10.19、PyTorch 2.8.0+cu128 和 torchvision 0.23.0+cu128。

```bash
conda env create -f environment.yml
conda activate re-tokendial
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./TokenDial -r TokenDial/evaluation/requirements.txt \
  timm==0.5.4 scipy huggingface-hub 'typing_extensions>=4.16'
python reproduce.py --help
```

外观训练需要 `timm==0.5.4`。上述安装方式使用评测依赖中的 `opencv-python-headless`，同一环境只装一种 OpenCV 包。Decord 0.6.0 在 Linux Python 3.10 上可能触发 `pip check` 的 wheel 标签告警。H20 验证用 Decord 抽帧检查了 18 条视频，均成功。

Conda 实测覆盖了「人物变老」单视频组的 24 次训练更新、两条 prompt 各三档的生成，以及六条新视频的评分一致性检查。验证环境复用了服务器已核验的部分运行包。上面的命令给出新机器上的安装方式，完整的全新依赖下载过程尚未从头实测。历史运行依赖见 [runtime-versions.txt](training/runtime-versions.txt)。

- 生成需要 Wan2.1-T2V-1.3B 的 DiT、UMT5、tokenizer 和 VAE 权重。可依照 [上游说明](TokenDial/README.md) 下载并缓存。四指标使用 `openai/clip-vit-base-patch32` 和 AlexNet LPIPS，离线环境需预先缓存。
- 外观训练准备有权访问的 `InternVideo2-stage2_1b-224p-f4.pt` 与 `bert-large-uncased` 本地模型目录。运动训练准备 DINOv2 源码及 torch hub 的 `dinov2_vitl14_pretrain.pth` 缓存。模型权重单独准备。原有 12 属性实验的训练视频按对应清单准备。
- [datasets/](datasets/) 包含本轮 12 个属性的 CSV，每行给出文件名、训练字幕和视频 SHA256。同目录的 sources.csv 给出原素材标识、可用来源链接和裁剪起点。将相应的预处理 MP4 放入同一个属性目录。精确复跑训练需要与清单哈希一致的文件；如果只有 checkpoint，可直接从第 3 步开始。

**2. 训练。** 以“人物变老”为例。将下列绝对路径换为实际位置。

```bash
JOB=person_older
CLIPS=/data/tokendial/person_older
export BASELINE_BERT=/models/bert-large-uncased
export INTERNVIDEO_CKPT=/models/InternVideo2-stage2_1b-224p-f4.pt
python reproduce.py verify "$JOB" --clips "$CLIPS"
python reproduce.py train "$JOB" --clips "$CLIPS"
```

本轮实际训练参数取自 [appearance.yaml](TokenDial/configs/appearance.yaml) 和 [motion.yaml](TokenDial/configs/motion.yaml)。外观用 InternVideo2 文本方向损失，832×480、17 帧、学习率 5e-4、2 个 epoch；运动用 DINOv2，672×384、33 帧、学习率 1e-4、2 个 epoch。训练运动时改用 `JOB=speed_higher_motion_magnitude` 和对应素材，并设置 `WAN_DINO_LOCAL=/path/to/dinov2`。`plan.json` 中旧的 `training` 段来自早期 300 步实验，本轮训练入口读取的是上述 YAML。训练集重建和 GPU 运算的差异都会影响结果，训练数值的逐位一致性尚未验证。

**3. 生成与四指标评分。**

```bash
python reproduce.py generate "$JOB"
python reproduce.py score "$JOB"
# 使用作者发布的人物权重进行同协议对照
python reproduce.py generate person_older --arm author --checkpoint /models/person_older.safetensors
python reproduce.py score person_older --arm author
```

每个属性使用 8 条公开 prompt 加 8 条本地补充 prompt，seed 均为 2。生成设置为 832×480、41 帧、24 fps、50 步、文本 CFG 4.5。外观五档为 0、0.25、0.5、0.75、1；运动加速沿 Wan 的负向 scale 取 0 至 −1。外观 mask 在前两步全局注入，第 2 步提取注意力，第 3 步起应用。两组共享零强度视频，每个外观属性、每套权重生成 144 个文件。运动使用首帧锚定和全局控制，两组共用 80 个文件。

默认自训输出目录为 `runs/$JOB/native/native/`，作者权重对照为 `runs/$JOB/author/author/`。批量复跑时对 `datasets/` 下各属性分别执行训练、生成、评分。比较时先检查 `alignment.json` 的 `passed`，再按有／无 mask 分组对照。摘要中区分逐样本 Overall 平均与先平均各分项再计算的 Overall。

| 指标 | 含义 | 方向 |
|---|---|---|
| CR | 目标属性的控制范围 | ↑ |
| Mono | 相邻强度档沿目标方向递进的程度 | ↑ |
| CSM | 相邻属性分数增量的标准差，衡量步幅是否均匀 | ↓ |
| LPIPS | 相对于零强度基准的感知差异 | ↓ |

本轮按帧步长 4、最多 16 帧评分，单调性容差 1e-4。Overall = CR / (1 + LPIPS) + (1 − CSM)。完整定义、输出字段和单独评分命令见 [四指标说明](TokenDial/evaluation/README.md)。

## 两种 VLM 评分如何运行

**作者 T2V 版。** 在环境中设置 `GEMINI_API_KEY`。下面的适配器把生成清单转为作者脚本需要的目录结构后调用原脚本。作者脚本默认模型为 `gemini-2.0-flash`；如果使用其他可用模型，应记录实际 ID。

```bash
ROOT="$PWD"
(cd TokenDial && python -m evaluation gemini \
  --manifest "$ROOT/runs/person_older/native/native/manifest.json" \
  --out-dir "$ROOT/runs/person_older/vlm_author" --model gemini-2.0-flash)
```

**项目 TV2V 版。** 建议使用独立的 Python 3.11+ 环境。复制示例清单、填写源视频和五个编辑输出路径，再运行以下命令。默认在三个对应时间点取帧；动态属性的评分受静帧证据限制。正式调用会将采样图像发送给 Gemini，API 密钥只从本地环境读取。

```bash
python -m pip install -r tv2v_vlm/requirements.txt
cp tv2v_vlm/manifest.example.json tv2v_vlm/manifest.json
# 填好 manifest.json 中的实际路径
python tv2v_vlm/gemini_tv2v_eval.py --manifest tv2v_vlm/manifest.json --dry-run
python tv2v_vlm/gemini_tv2v_eval.py --manifest tv2v_vlm/manifest.json --out-dir runs/vlm_tv2v
```

## 来源

[TokenDial 论文](https://arxiv.org/abs/2603.27520) · [上游仓库，本轮记录的版本](https://github.com/ariannaliu/TokenDial/tree/711d6b7749b29aeb9e307a65012daf9d5d495ce0) · [上游许可](TokenDial/LICENSE) · [第三方声明](TokenDial/NOTICE)。四指标和原版 VLM 的作者提供文件及其校验值见 [provenance.json](TokenDial/evaluation/vendor/provenance.json)。本项目另提供评估适配器、复现协议和 TV2V 评分脚本。

