# re-tokendial · Wan 训练、生成与评分

## 最新实验 · 人物变老，1 条与 8 条训练视频

两组都使用 Wan2.1-T2V-1.3B 和 TokenDial 的人物变老方向。固定初始化种子 `20261123`、AdamW、学习率 `5e-4`、相同学习率调度和 **24 次参数更新**。训练裁切设置为 832×480、17 帧。单视频组每轮重复同一条两次，共 12 轮。8 视频组每轮各使用一次，共 3 轮，并包含单视频组用过的那条。

训练素材的文件名、caption、原始来源及 SHA256 见 [训练清单](experiments/person_older/metadata.csv)。把与清单哈希匹配的预处理 MP4 放入同一目录。单视频组只需清单第一行所列的 `clip_00001.mp4`。环境和权重准备见下文，训练使用已分配的单张 CUDA GPU。

```bash
export BASELINE_BERT=/models/bert-large-uncased
export INTERNVIDEO_CKPT=/models/InternVideo2-stage2_1b-224p-f4.pt
CLIPS=/data/tokendial/person_older

for GROUP in single multi; do
  python reproduce.py verify person_older --study "$GROUP" --clips "$CLIPS"
  python reproduce.py train person_older --study "$GROUP" --clips "$CLIPS"
  python reproduce.py generate person_older --study "$GROUP"
  python reproduce.py score person_older --study "$GROUP"
done
```

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | 视频、caption 和外观训练权重 | `training/checkpoints/`、`step_loss.csv`、`loss_curve.json`、恢复断点 |
| 推理 | 第 24 次更新的 checkpoint、冻结的 prompt | `unseen6/native/nomask/` 下 18 个 MP4，以及含每条视频哈希的 `manifest.json` |
| 评测 | 推理清单、CLIP、AlexNet LPIPS 和人物变老 guidance | `metrics/metrics.csv`、`metrics/summary.json`、`alignment/alignment.json` |

默认输出根目录为 `runs/person_older/matched/<single或multi>/`。已有 checkpoint 时可直接执行 `generate`，通过 `--checkpoint /path/to/weights.safetensors` 指定文件。默认 checkpoint 分别是 `training/checkpoints/epoch-11.safetensors` 和 `training/checkpoints/epoch-2.safetensors`，文件名从 0 开始。训练途中重跑同一命令会使用断点。更换 checkpoint 后需使用独立的 `--work` 目录。

[推理协议](experiments/person_older/protocol.json) 固定六条未参与训练的 prompt、seed 2、外观强度 0 / 1 / 3、无 mask、832×480、41 帧、24 fps、50 个采样步和文本 CFG 4.5。要复跑初次检查的原 caption 和咖啡馆 prompt，在生成和评分时都添加 `--prompt-set smoke`。早期单视频 checkpoint 可用 `--checkpoint` 指定 `epoch-1.safetensors`，并用独立工作目录存放结果。

对比时按同一个场景和 seed 对齐两组 `metrics.csv`，确认各自的 `alignment.json` 通过，再比较 CR、Mono、CSM、LPIPS、Overall，并查看三档的年龄递进和人物身份。这里的六组对照使用三档和一个 seed，论文复现实验的五档协议见下文。

## 原有 12 属性 Wan 复现实验

本仓库整理了 2026 年 11 月这一轮 TokenDial **公开 Wan 实现**的复现实验。实验从 Table 7 的 12 个概念中各选一个属性，自训视觉 token 偏移权重，再用相同 prompt、seed 和五档强度生成视频。论文 Table 2 使用另一套主实验骨干，并汇总更多属性；这里可以复跑我们的 Wan 实验，并与论文报告值作参考对照。

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

**1. 准备运行环境与输入。** 本轮训练环境为 Python 3.10、PyTorch 2.8.0、torchvision 0.23.0。先安装与本机 CUDA 驱动兼容的 PyTorch，再安装项目依赖。服务器的版本记录见 [runtime-versions.txt](training/runtime-versions.txt)。

```bash
python -m pip install -e './TokenDial[train]' -r TokenDial/evaluation/requirements.txt
python reproduce.py --help
```

- 生成需要 Wan2.1-T2V-1.3B 的 DiT、UMT5、tokenizer 和 VAE 权重。可依照 [上游说明](TokenDial/README.md) 下载并缓存。四指标使用 `openai/clip-vit-base-patch32` 和 AlexNet LPIPS，离线环境需预先缓存。
- 外观训练准备有权访问的 `InternVideo2-stage2_1b-224p-f4.pt` 与 `bert-large-uncased` 本地模型目录。运动训练准备 DINOv2 源码及 torch hub 的 `dinov2_vitl14_pretrain.pth` 缓存。权重和视频单独准备。
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

