# 阿里云模型选型与项目接入

核对日期：2026-09-11。以下按能力覆盖千问与阿里自有媒体模型的主要系列，
列出本项目相关型号及其他系列入口，不把历史快照逐个重复列出。
完整、动态的地域型号清单见[百炼模型广场](https://bailian.console.aliyun.com/cn-beijing?tab=model#/model-market)。
官网列有模型不等于当前账号已经获得调用权限。

## 一个 Key 怎样调用多个模型

百炼 Key 归属于业务空间，可以调用该空间获授权的多个模型；不需要为文生文、
文生图、视频生成分别建立 Key。地域、业务空间和接入地址必须匹配。
这里的 API 是百炼服务，不是千问聊天网页的登录凭证或网页会员权益。
依据：[API Key](https://help.aliyun.com/zh/model-studio/get-api-key)、
[权限说明](https://help.aliyun.com/zh/model-studio/application-permission-management-overview)。

项目用一个 `.env` 条目 `DASHSCOPE_API_KEY`，三个 provider 都引用这个环境变量名。
模型名和生成参数分别放在 YAML 的 `vlm`、`image_generation`、`video_generation`。
切换模型通常无需换 Key，但适配器必须支持该模型的输入输出协议。

## 模型系列全景

| 类别 | 主要系列或型号 | 对本项目的意义 |
| --- | --- | --- |
| 通用文本、推理与视觉 | `qwen3.8-max`、`qwen3.8-max-0902`、`qwen3.8-flash`、`qwen3.7-plus`、`qwen3.7-flash` | 生成提示词、理解首帧、输出 QC JSON；本次选 Max |
| 专门视觉理解 | `qwen3-vl-plus`、`qwen3-vl-flash`；旧版 Qwen-VL、QVQ | 可作为后续 QC 对照，不能拿视觉理解模型名称直接生成图片 |
| 开源权重系列 | Qwen3.8、Qwen3.6、Qwen3.5、Qwen3、Qwen2.5 及不同参数规模 | 部分可在百炼调用，也可研究自部署；模态和参数随具体版本变化 |
| 编程与历史文本型号 | Qwen3-Coder、Qwen-Long、旧版 Max/Plus/Turbo、Qwen-Math、QwQ | 文本或专用任务；不默认具备视频质检所需能力 |
| 千问图像生成、编辑 | Qwen Image 3.0 Pro/3.0、2.0 Pro/2.0、Image Max/Plus、Image Edit Max/Plus | 初始场景图；本次接入 3.0 的文生图路径 |
| 其他阿里图像生成 | Wan 2.7 Image Pro/Image、Z-Image Turbo | 图像任务备选，尚未实现这些协议 |
| 万相视频 | Wan 3.0 Video/Video Prime；历史 Wan 2.7/2.6/2.5/2.2/2.1 的 T2V、I2V、R2V 等型号 | 本次接入 Wan 3.0 首帧生视频；不同代际不能直接互换请求字段 |
| 其他阿里视频模型 | HappyHorse 1.1 T2V/I2V/R2V、1.0 Video Edit | 视频生成、参考与编辑的后续对照，未接入 |
| 全模态理解、实时交互 | Qwen3.5-Omni Plus/Flash 及 Realtime | 音视频交互；V1 QC 目前只检查视觉帧 |
| 语音与音乐 | Qwen Audio 3.0 TTS/ASR/Realtime、Fun 系列语音/音乐 | 语音合成、识别、对话、音乐；不进入本次质检链路 |
| 检索向量与重排序 | Qwen3.7 Text Embedding、Qwen3-VL Embedding、Qwen3.7 Text Rerank | 检索和相似度服务；不引入为 V1 的额外质量打分器 |

分类依据：[总目录](https://help.aliyun.com/zh/model-studio/models)、
[文本模型](https://help.aliyun.com/zh/model-studio/text-generation-model)、
[视觉模型](https://help.aliyun.com/zh/model-studio/vision-model/)、
[图像模型](https://help.aliyun.com/zh/model-studio/image-model)、
[视频模型](https://help.aliyun.com/zh/model-studio/video-generate-edit-model/)。
百炼也托管第三方品牌；本次组合全部选择阿里模型。

## 本次组合

| 环节 | 模型 | 请求方式 |
| --- | --- | --- |
| 初始图提示词、看首帧写视频提示词、独立 QC | `qwen3.8-max` | `/compatible-mode/v1/chat/completions`，每次独立 system/user，多图输入，QC 使用 JSON mode |
| 初始场景图 | `qwen-image-3.0-pro` | `/api/v1/services/aigc/multimodal-generation/generation`，单张图同步返回 |
| 首帧生视频 | `wan3.0-video` | `/api/v1/services/aigc/video-generation/video-synthesis`，提交后按 task ID 轮询 |

选型理由是三段能力齐全、模型 ID 与官方 API 明确，并能沿用当前北京 Key。
Qwen3.8-Max 支持图像、视频输入及结构化输出，适配当前逐帧证据检查方式。
图片与视频都关闭服务端自动改写，以保留已保存提示词与生成请求之间的对应关系。
这不是这些模型优于其他模型的实证结论；一次连通性试跑也不能验证 QC 准确率。
参考：[Qwen3.8-Max](https://help.aliyun.com/zh/model-studio/qwen3-8-max)、
[图像生成 API](https://help.aliyun.com/zh/model-studio/qwen-image-generation-and-editing-api-reference)、
[Wan API](https://help.aliyun.com/zh/model-studio/wan3-video-generation-api-reference)。

使用 `configs/aliyun-beijing.yaml`：三段共享 Key，生成 1 张 1024×1024 图、
1 段 720P / 5 秒视频，均匀抽取 16 帧做 QC。必须显式传入 `--allow-paid`。
默认不指定配置时仍为离线 mock。原 `configs/qwen-beijing.yaml` 仍是 VLM 接口配置。

北京原价参考：Qwen3.8-Max 输入 12 元/百万 Token、输出 36 元/百万 Token；
Qwen Image 3.0 Pro 的 1K 输出约 0.25 元/张；Wan 3.0 的 720P 原价 0.6 元/秒。
因此单次图像加 5 秒视频按原价约 3.25 元，另加三次 VLM 消耗。
免费额度、缓存、限时折扣和账号计费状态会改变实付；以控制台账单为准。
依据：[价格表](https://help.aliyun.com/zh/model-studio/model-pricing)。

## 接入边界与恢复

已实现的媒体型号是 `qwen-image-3.0-pro` / `qwen-image-3.0` 与
`wan3.0-video` / `wan3.0-video-prime`。目录中的其他型号只是调研备选，
不能理解为都已接入或已用当前 Key 实测。未做批量模型试跑。

生成失败不会代换成 mock 或重新生成。Wan 的任务 ID 会立即落盘；可通过
`WanVideoGenerator.resume` 取回已有任务，再用 `judge` 独立质检，详见 README。
QC 始终只获得原始任务和视觉证据；生成模型返回的自述、隐藏物理状态、机器人执行
结果不会进入评判。多个阶段共用同一模型家族可能存在共同偏差，应在后续人工标注
样本上分别评估误拒、漏检与不确定比例。

## 本次真实验证

2026-09-11，在北京域名使用同一个本地 Key 完成一次全流程：
Qwen3.8-Max 提示词 → Qwen Image 3.0 Pro 首帧 → Qwen3.8-Max 视频提示词 →
Wan3.0-Video → Qwen3.8-Max 独立 QC。共生成 1 张图和 1 段视频，没有重提生成任务。
耗时约 171 秒，首帧为 1024×1024 PNG；视频请求使用 720P 档位和自适应比例，
实际返回 960×960、5 秒、30 FPS、150 帧、无音轨。QC 抽取 16 帧，三个检查均为
`pass`，Python 汇总为 `PASS`，`is_mock=false`。这是单个样本的模型判断，
不能视为准确率评估、全部时刻检查或物理可执行性认证。

同日 `/compatible-mode/v1/models` 返回 249 个型号，按千问、万相、HappyHorse、
Z-Image 和阿里语音系列前缀筛出 183 个。这个列表表示接口返回的目录，不能证明
每一项都已获当前 Key 授权；本次只实测上述三个型号。目录 JSON 和完整运行产物
保存在本次会话的输出附件中，未将 Key 或签名媒体链接写入附件。
