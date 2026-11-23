# TV2V Gemini 评分

使用 Python 3.11 或以上版本。默认评分模型为 Gemini 3.8 Flash，支持用 `--model` 显式指定。固定一个模型 ID 可让同一批实验使用相同版本。

```powershell
python -m pip install -r requirements.txt
Copy-Item manifest.example.json manifest.json
python gemini_tv2v_eval.py --manifest manifest.json --dry-run
```

先编辑 `manifest.json`，把源视频和五个输出路径指向本地文件。支持相对 manifest 的路径以及绝对路径。每条样本按 `-4、-2、0、2、4` 排列。`prompt` 可填该档的编辑指令，用同一 slider 生成时可省略。源视频与输出使用同一播放起点，建议采用恒定帧率的视频。

正式评分前在当前终端设置 `GEMINI_API_KEY`。密钥只由环境变量读取，目前留空。运行正式评分时，采样图像会发送给 Gemini。

```powershell
python gemini_tv2v_eval.py --manifest manifest.json --out-dir vlm_run
# 中断后用同一命令追加 --resume，复用已成功的时间点。
```

## 评分协议

- 默认在六个视频的共同时间段均匀取三个时间点。帧率一致时，沿用作者在最短视频帧数范围均匀取帧索引的逻辑。帧率不同时，按播放秒数找各视频最接近的帧。
- 每个时间点提交一张外部源视频图和五张输出图，输出依次标为 `-4、-2、0、2、4`。前三项及强度递进沿用作者的四个维度和 0–5 分量表。递进衡量目标属性沿有符号档位的变化。
- 源视频保持以外部输入为参考，考察目标属性之外的改动。0 档为源视频原文件或字节相同的副本时自动从本项中排除。若 0 档是人工转码得到的源片副本，可显式设 `zero_is_source_copy` 为 `true`。实际由模型生成的 0 档也参与本项。
- `scores.csv` 给出逐样本五项均分及 `overall_4`。四项均分保留作者的汇总口径，源视频保持单列。`scores.jsonl` 保留逐时间点的五项分数、理由、原始响应和帧索引。`protocol.json` 记录模型、采样数、脚本、输入清单和视频文件哈希。
- 本版沿用作者的同时间点静帧输入。速度、频率、闪烁等属性需要时序证据，评分理由应标明静帧证据的局限。TV2V 输入、双向档位和新版 Gemini 均应记录在实验设置中。

参考 [TokenDial 论文附录 A.2](https://arxiv.org/abs/2603.27520) 和 [Gemini 3.8 Flash 文档](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)。
