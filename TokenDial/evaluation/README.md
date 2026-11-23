# TokenDial 评测接入与作者代码对齐

本模块放在 TokenDial 仓库根目录的 `evaluation/`。本轮目标是评测实现及输入接口一致，不是复现论文主表数值。适用于已有多档结果视频；评测本身不需要加载 Wan 生成模型。

## 代码来源与边界

- `vendor/evaluate_sliders.py`、`vendor/gemini_slider_video_eval.py` 是用户从作者取得的原始文件，按字节复制，未修改。SHA-256 见 `vendor/provenance.json`。
- `guidance/` 保留作者提供的 56 份配置，`prompts/` 保留 13 份场景文本。
- 新入口复用作者的 CLIP logits、视频抽帧和 LPIPS 预处理函数。清单读取、数据校验、四项指标归约及汇总是新增代码。
- 对齐检查另行运行完整、未改动的作者脚本，再比较 CSV。由于编码器和低层处理复用，对齐证明的是接口及归约未改变原协议，不是独立重写编码器后的交叉验证。
- 原文件的来源归属作者；此本地集成不主张改变其许可，发布或再分发前应确认这些单独提供文件的授权。

## 环境

服务器验证环境：Linux x86_64，Python 3.10（满足仓库 >=3.10），独立 venv。使用固定版本 `torch==2.8.0`、`torchvision==0.23.0`，安装与驱动匹配的 CUDA 构建；其余评测依赖见 requirements.txt。推理依赖还需按 TokenDial 的 pyproject.toml 安装。

```bash
python -m pip install -r evaluation/requirements.txt
python -m unittest discover -s evaluation/tests -v
python -m evaluation --help
```

`opencv-python-headless` 提供 cv2，服务器不需要桌面显示。`google-genai` 只在调用 Gemini 时使用，四项自动指标不依赖 API 密钥。视频生成和评测建议分进程运行，避免同时占用生成模型和评价模型的显存。

## 标准输入

每条样本是一条同任务、同输入、同 seed 的强度轨迹。最少两档，VLM 固定五档。单个编辑结果不能计算多档单调性与 CSM，当前入口会拒绝，不能把单个结果虚构成多个档位。

```json
{
  "schema_version": 1,
  "generation": {"repository_commit": "固定的提交", "checkpoint": "权重路径"},
  "samples": [{
    "method": "tokendial", "concept": "person", "attribute": "older",
    "sample": "seed_0", "seed": 0, "prompt": "A man ...",
    "source_video": "source.mp4",
    "baseline_kind": "zero_strength_reconstruction",
    "clips": [
      {"intensity": 0, "model_scale": 0, "is_baseline": true, "video": "base.mp4"},
      {"intensity": 1, "model_scale": 2, "is_baseline": false, "video": "edit.mp4"}
    ]
  }]
}
```

路径相对于 manifest 所在目录解析。`intensity` 是严格递增的逻辑顺序，`model_scale` 是实际传给生成模型的强度，二者不要混淆。第一个逻辑档必须显式标为基准。例如“更快”可以使用真实强度 `0,-1,-2,-3,-4`，逻辑顺序仍为 `0,1,2,3,4`，不能按负数数值排序后把 -4 当原视频。

原始 source_video 只用于追溯，不会自动替代基准。当前四项指标以 clips 中指定的基准计算；若是零强度 VAE 重建，结果不等于相对原始视频的保持分数。

## 从 TokenDial 输出创建清单

默认接受 `videos/seed_0/flowedit_rgb0.0.mp4` 至 `flowedit_rgb4.0.mp4` 等 V2V 文件名；T2V 使用 `--kind t2v`，接受 `cfg_0.0.mp4` 等。

```bash
python -m evaluation adapt --root /path/to/videos --manifest /path/to/manifest.json \
  --seeds '0,1' --model-scales '0,1,2,3,4' --concept person --attribute older \
  --source-video assets/demo_videos/man_real.mp4 \
  --checkpoint checkpoints/person_older.safetensors --prompt 'A man ...'
python -m evaluation validate --manifest /path/to/manifest.json \
  --guidance-dir evaluation/guidance --probe
```

也可直接按格式编写清单以接入其他 TV2V 方法。`--probe` 校验帧数、分辨率、帧率相同；正常评价会强制执行此检查，避免帧配对含义不明。缺失文件、重复样本、重复强度、错误基准及缺 guidance 均报错，不静默跳过。

## 评价与对齐

```bash
python -m evaluation evaluate --manifest /path/to/manifest.json \
  --guidance-dir evaluation/guidance --out-dir /path/to/run/candidate --device cuda
python -m evaluation align --manifest /path/to/manifest.json \
  --guidance-dir evaluation/guidance --metrics-csv /path/to/run/candidate/metrics.csv \
  --out-dir /path/to/run/alignment --device cuda
```

两个命令必须使用相同的 `--stride`、`--max-frames`、batch size 和 `--monotonicity-eps`。默认 stride=4、最多16帧、两个batch size均为8、eps=1e-4。CLIP 固定 `openai/clip-vit-base-patch32`，LPIPS 固定 AlexNet。使用原始 CLIP logits，CSM 为相邻差值的 float32 总体标准差，尺度间距不进入公式。

输出目录必须为空，避免混入旧结果。输出包括：

- `metrics.csv`：逐轨迹四项指标和新增的样本级 Overall。
- `features.jsonl`：逐档 CLIP/LPIPS 原始分数、输入清单和视频 SHA-256。
- `summary.json`：按方法作样本微平均，分别报告“样本 Overall 的均值”及“用均值指标计算的 Overall”。不将任一种冒充已验证的论文汇总规则。
- `run.json`：运行状态、配置、包版本、仓库提交、作者代码及 guidance 哈希、CLIP 权重提交标识。
- `alignment/author_*.csv`、日志及 `alignment.json`：完整作者脚本结果和对比。

验收要求是同样本集合、同 anchor 和档位，四项指标在 `atol=1e-5, rtol=1e-5` 内一致。缺行、多行、NaN、文字配置不一致都会失败。Overall 的公式为 CR/(1+SP)+(1-CSM)，只验证公式与记录规则，没有作者汇总代码可作比对。

## 小规模端到端验证

```bash
export CUDA_VISIBLE_DEVICES=分配给你的GPU编号或UUID
export ARTIFACT_DIR=/path/to/a/new/empty/run
bash evaluation/scripts/run_validation.sh
```

脚本使用人物变老滑块、man_real.mp4、seed 0和1，每条五档。为验证流程选择17帧、320×576、50步、n_max=40、无掩码。它不是论文完整评测集，不用于宣称论文得分复现。视频、环境锁定文件、生成日志和清单保存在仓库同级 `artifacts/person_older_v2v/`。

若 Hugging Face 主站不可达，可显式设置 HF_ENDPOINT。模型镜像来源、固定revision、文件大小及SHA校验应随下载记录保存。不要把访问密钥写进清单、源码或日志。

## 可选 Gemini

```bash
# 通过运行环境安全设置 GEMINI_API_KEY 后执行；请勿把实际密钥写入脚本。
python -m evaluation gemini --manifest /path/to/manifest.json \
  --out-dir /path/to/vlm_run --model 你确认可用的模型ID
```

默认取3个共同时间位置，每次把五档同一时刻的图像交给作者原始 rubric。评价 prompt following、identity、background、progressive intensity；最后一项是跨强度渐进性，不是整段视频的时间连续性。要求显式指定模型，不沿用未经验证的历史模型默认值。未配置 API 时不能宣称 VLM 在线验证通过。

## 当前范围之外

PickScore、ViCLIP、独立时序连续性、区域掩码评价、完整论文任务集及重新训练均未纳入本轮。当前样本级检查保证数据一致，不代表评测集具有统计代表性。
