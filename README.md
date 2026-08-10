[English](README.en.md) · 中文

# ChatRouter

面向生产环境、兼容 OpenAI 协议的 LLM 流量网关，专注于**智能请求路由**与**精细化流量治理**。

作为 OpenAI SDK 的 drop-in 替换，ChatRouter 在单一入口后面统一管理多家供应商的模型池，在保障服务质量的前提下持续压降推理成本。

---

## 两大核心能力

### 1. 全对话上下文感知路由

朴素**单轮路由器（single-turn router）**只对**最后一轮用户提问**做难度判断，这在真实多轮对话中会系统性失准：

| 场景 | 单轮路由的判断 | 实际需求 |
|------|--------------|---------|
| `"那另一种情况呢？"` | 极简短 → 廉价模型 | 继承前文的复杂推导要求 |
| system 提示要求"所有回答必须给出严格证明" | 后续提问看似简单 | 全程需要强推理能力 |
| 用户连续三次回复"还是不对" | 每轮独立看都很短 | 当前档位已被证伪，必须升档 |
| 长达 40 轮的架构讨论 | 末轮只是一句追问 | 需同时满足累积的全部约束 |

> 注意：上述失准是**单轮路由**的局限，而非"对话感知路由"的通病。MTRouter、Router-R1、LLMRouter 等对话感知方案同样以"看历史"为核心设计目标——区别在于它们多用学习的**历史-模型联合嵌入**来表征上下文，而 ChatRouter 用**可解释的规则信号 + 近因加权**达到同样目标，无需训练数据、决策可逐条审计。

ChatRouter 对**完整对话历史**打分，通过三个机制避免上下文任务错配：

- **近因加权**：越近的轮次权重越高（`recency_decay`），但历史轮次始终保留影响力，早期声明的硬性要求不会被遗忘。
- **升级记忆**（`escalation_memory`）：一旦某个线程展现出高难度特征，加权均值会向历史峰值回拉——难任务不会因为一句"好的"就被悄悄降档。
- **指代继承**：识别 `"继续"`、`"那另一个呢"` 这类无独立语义的追问，直接继承前文的复杂度。

十个信号维度参与打分，中英双语识别：推理关键词、代码含量、对话深度、上下文压力、未解决线程（用户不满信号）、工具调用、结构化输出、指令密度、期望输出长度、多语言混合。

> **聚合方式**：得分并非简单加权平均。多数信号在单次请求中为零，纯均值会把"证明该定理"这类明确的强信号稀释到廉价档。因此实际采用 `主导信号 × 0.6 + 加权均值 × 0.4 + 交互项`，既保证单个决定性信号能主导决策，也让"多个中等信号叠加"同样被识别为高难度。

`/v1/routing/explain` 可直接查看完整决策依据，无需真实调用模型。

### 2. 在线反馈闭环自适应路由

路由策略不会停留在配置文件写死的先验值，而是依据**真实线上数据**持续自我迭代：

- **显式反馈**：`/v1/feedback` 接收点赞/点踩、1–5 星评分、采纳与否。
- **隐式信号**：重试次数、输出截断（`finish_reason=length`）、相对基线的延迟劣化、上游失败，均自动折算为质量证据。
- **分档统计**：统计按 `(模型, 复杂度档位)` 分别累积。某模型可能在简单任务上表现优异、在强推理任务上明显偏弱——全局均值会掩盖这一差异，分档统计不会。
- **置信度加权**：样本越多，实测质量对先验的覆盖越强；仅有两条评分的模型几乎不影响决策，积累数百条后则占据主导。
- **单次生效**：每个 `request_id` 只能被评分一次，重复提交会被丢弃（`accepted: false`）。由于 `request_id` 会通过响应头返回给客户端，若允许重放，任何持有它的调用方都能对某个模型灌入任意多次差评，把它挤出候选池——这是自适应路由的投毒入口，必须在入口处堵死。
- **证据半衰**：超出统计窗口的历史证据按指数衰减，上周的一次故障不会永久拖累该模型。
- **探索机制**：以 `exploration_ratio` 的概率采样次优候选，避免策略锁死在局部最优、让长尾模型持续产生可用证据。

效果：某模型在特定任务类型上质量下滑时，网关会在数十次请求内自动减少对它的调度，无需人工介入改配置。

#### 反馈归一化

客户端表达满意度的"语言"很多样：直接给 `score`、打 1–5 星 `rating`、点 👍/👎、给出 `accepted`/`regenerated`/`edited` 等行为信号。`/v1/feedback` 会把这些**不同形态统一折叠成 `[0,1]` 质量分**后再进入统计闭环，折叠逻辑集中在可配置项 `routing.feedback.normalization` 中：

| 形态 | 默认映射 |
|------|----------|
| `score` | 原值（已在 `[0,1]`） |
| `rating` | `(rating-1)/4`，1→0.0，5→1.0 |
| `thumb: up/down` | `1.0` / `0.0` |
| `accepted: true/false` | `1.0` / `0.0` |
| `regenerated` | `0.2`（弱负面，重生成说明首次不满意） |
| `edited` | `0.5`（中等负面，答案有用但需改动） |

优先级为 `score > rating > thumb > accepted > regenerated > edited`：评分越"刻意"，权重越高。归一化结果会在反馈响应里通过 `source` 字段回显原始信号，便于审计"这条分数从哪来"。把映射做成配置（而非写死在请求 schema 里）的意义在于：运营方无需改代码即可按业务调优，且每条反馈的可解释性始终可追。

---

## 流量治理

| 能力 | 说明 |
|------|------|
| **RPM / TPM 限流** | 租户级请求数与 token 双维度限流。token 先按估算预扣，响应返回后用真实用量对账回补 |
| **并发控制** | 租户级在途请求数上限 |
| **租户配额** | 小时/天/月窗口的请求数、token、美元花费三维配额，超限支持 `reject`（拒绝）或 `downgrade`（降级到最廉价档继续服务） |
| **负载溢出调度** | 模型饱和时自动溢出到有余量的替代模型；全部饱和时可短暂排队等待而非直接失败 |
| **故障降级** | 熔断器隔离故障上游（closed → open → half-open 自动探活），失败请求沿 fallback 链降级重试 |
| **重试策略** | 指数退避 + 抖动；4xx 客户端错误不重试（换模型也必然失败）；流式响应仅在首字节送达前可重试 |
| **上下文超限兜底** | 对话超出所有候选模型窗口时不再直接失败，按 `routing.context_overflow.strategy` 降级处理 |

### 上下文超限策略

长对话超出窗口在生产中是常态而非异常，因此需要有明确的降级路径：

| 策略 | 行为 |
|------|------|
| `reject` | 直接返回 HTTP 400（`context_length_exceeded`）。重试和扩容都无济于事，让客户端尽早知道真相 |
| `largest_window`（默认） | 路由到窗口最大的模型，即使其档位与任务复杂度并非最佳匹配 |
| `trim_history` | 裁剪对话中段使其放得下。**有损**，故需显式开启 |

裁剪遵守两条不变量：**system 提示词**与**最近若干轮**永不丢弃——前者决定模型的行为约束，后者才是真正被提问的内容，丢掉任一个都会让模型悄悄回答另一个问题。中段按由旧到新的顺序移除，并默认插入省略标记，避免模型把剩余轮次误读为连续对话。

### 响应缓存

对**非流式**且各字段完全一致的请求，命中缓存后直接返回上次结果，**不再调用上游**，从而省下一次生成开销。配置项 `routing.response_cache`：

| 字段 | 说明 |
|------|------|
| `enabled` | 默认关闭；开启后才会参与缓存 |
| `ttl_seconds` | 缓存条目存活时间（默认 300 秒） |
| `bypass_hints` | 带有这些 `chatrouter` 提示词的请求跳过缓存，默认 `["session_id"]`——多轮对话依赖缓存看不到的上下文状态 |
| `excluded_tenants` | 这些租户永远不读不写缓存（如必须每次都触达模型的审计/评测租户） |

缓存键包含**所有会影响生成文本的字段**：解析后的目标模型、`messages` 全量、以及 `temperature` / `top_p` / `max_tokens` / `stop` / `tools` / `tool_choice` / `response_format` / `seed` / `user` / `n` 等采样参数。任一不同都视为不同请求，避免把 A 的答案错发给 B。流式响应因 SSE 实时投递而从不缓存。

命中缓存时仍会执行完整的记账流程（配额、计费、反馈学习），因此缓存命中对网关其余部分与一次真实生成**不可区分**——只是不再产生上游费用。命中时响应头带 `x-chatrouter-cache: HIT`。


若受保护的首尾部分本身就超出预算，裁剪不会假装成功，而是在决策 `notes` 中如实记录。

---

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

cp config/config.example.yaml config/config.yaml
# 编辑 config.yaml，配置你的 provider 与模型池

export OPENAI_API_KEY=sk-...
export CHATROUTER_ADMIN_KEY=your-admin-key

python -m chatrouter --check    # 校验配置
python -m chatrouter            # 启动服务
```

Docker：

```bash
docker compose up -d
```

### 调用方式

将 `base_url` 指向网关即可，其余与 OpenAI SDK 完全一致：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-chatrouter-dev")

# model="auto" 触发智能路由
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "请推导该递归的时间复杂度并证明其最优性"}],
)

print(response.model)  # 实际承接请求的模型
```

提交反馈以驱动策略迭代（`request_id` 取自响应头 `x-chatrouter-request-id`）：

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Authorization: Bearer sk-chatrouter-dev" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "chatcmpl-rt-...", "thumb": "down"}'
```

查看路由决策全过程（不实际调用模型）：

```bash
curl -X POST http://localhost:8000/v1/routing/explain \
  -H "Authorization: Bearer sk-chatrouter-dev" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"证明这个定理"}]}'
```

---

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 对话补全（流式 / 非流式），OpenAI 兼容 |
| GET | `/v1/models` | 模型列表（按租户权限过滤） |
| GET | `/v1/models/{id}` | 单个模型详情 |
| POST | `/v1/feedback` | 提交质量反馈 |
| POST | `/v1/routing/explain` | 路由决策试算 |
| GET | `/healthz` · `/readyz` | 存活 / 就绪探针 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/admin/status` | 实时负载、熔断、配额、学习到的质量 |
| GET | `/admin/config` | 生效配置（凭据已脱敏） |

### 响应头

| 头部 | 含义 |
|------|------|
| `x-chatrouter-request-id` | 请求 ID，提交反馈时使用 |
| `x-chatrouter-model` | 实际承接的模型 |
| `x-chatrouter-routing-reason` | 路由原因（`context_aware` / `feedback_adaptive` / `overflow` / `exploration` …） |
| `x-chatrouter-complexity` | 对话复杂度得分 `[0,1]` |
| `x-chatrouter-tier` | 判定的能力档位 |
| `x-chatrouter-context-trimmed` | 为适配窗口而裁剪掉的消息条数（仅在发生裁剪时出现） |
| `x-chatrouter-cache` | 命中响应缓存时为 `HIT`，未命中/绕过时不出现 |
| `x-ratelimit-*` | 限流余量 |

### 请求级路由提示

在请求体中附加 `chatrouter` 字段可对单次请求微调策略：

```json
{
  "model": "auto",
  "messages": [...],
  "chatrouter": {
    "session_id": "conv-123",
    "min_tier": "premium",
    "prefer_models": ["gpt-4o"],
    "quality_bias": 0.9
  }
}
```

---

## 配置要点

模型池按能力分为四档：`economy` → `standard` → `premium` → `reasoning`。复杂度得分经阈值映射到目标档位，再由效用函数在档内择优：

```
utility = 质量偏好 × 反馈修正质量
        + 成本偏好 × 成本得分
        + 延迟偏好 × 延迟得分
        + 负载得分
        - 档位偏移惩罚
        - 健康度惩罚
```

`routing.quality_bias` 是核心调节旋钮：`0` 为极致省钱，`1` 为极致质量，默认 `0.6`。可按租户覆盖。

完整配置说明见 `config/config.example.yaml`，所有字段均有注释。支持 `${VAR}` 与 `${VAR:-默认值}` 环境变量展开。

---

## 部署形态

- **单副本**：`storage.backend: memory`，零外部依赖。
- **多副本**：`storage.backend: redis`。限流计数、配额与反馈统计经 Redis 共享（计数器使用 Lua 脚本保证原子性），网关本身无状态，可水平扩展。熔断器状态刻意保持进程内——每个副本对故障的观测本就一致，本地决策更快。

---

## 开发

```bash
pip install -e ".[dev]"
pytest              # 157 个测试
ruff check src tests
```

测试覆盖：复杂度分析（含上下文感知的关键断言）、路由决策、反馈自适应、限流、配额、熔断、负载溢出、配置校验，以及基于 mock 上游的端到端 HTTP 流程（含流式与故障降级）。

---

## 范围说明

本项目**仅聚焦后端 LLM 流量治理**，不包含：智能体工作流编排、RAG 检索编排、模型训练与微调、前端交互界面。

## 许可

MIT
