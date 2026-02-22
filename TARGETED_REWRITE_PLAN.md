# 定向重写清单（面向实习投递）

> 目标：不是全量重写仓库，而是用最短时间拿到“独立实现 + 可量化优化”证据。

---

## 1. 重写范围（只做高价值部分）

在 `rewrite_lab/` 新建你的实现：

```bash
mkdir -p rewrite_lab/include rewrite_lab/src
```

只重写这 4 个文件：

1. `rewrite_lab/include/pipeline.h`
2. `rewrite_lab/src/pipeline_cpu.cpp`
3. `rewrite_lab/src/main.cu`
4. `rewrite_lab/src/pipeline_gpu.cu`

不重写：

- 原仓库 `include/` 和 `src/`（保留作参考）
- 现有 `Makefile`（先复用，后续再加 rewrite target）

---

## 2. 固定顺序（不要跳步）

### Step A：接口先行（半天）

先写 `rewrite_lab/include/pipeline.h`，定义：

- `PipelineConfig`
- `PipelineOutputs`
- `GpuRunStats`
- `run_cpu_pipeline`
- `run_gpu_pipeline_v1`

验收：`main.cu` 能包含头文件并通过编译。

### Step B：CPU baseline（1 天）

写 `rewrite_lab/src/pipeline_cpu.cpp`：

- `add_bias + relu`
- `row_sum`
- `normalize`
- `row_argmax`

验收：小数据人工可验证，输出稳定。

### Step C：主程序骨架（半天）

写 `rewrite_lab/src/main.cu`：

- 参数解析（batch/dim/warmup/iters/seed）
- 随机输入生成
- 调 CPU pipeline 并打印结果摘要

验收：程序可独立跑通 CPU 路径。

### Step D：GPU V1（1~2 天）

写 `rewrite_lab/src/pipeline_gpu.cu`：

- `add_bias_relu_kernel`
- `row_sum_kernel`
- `normalize_kernel`
- `row_argmax_kernel`
- `run_gpu_pipeline_v1`（含 cudaEvent 计时）

验收：CPU/GPU 对齐（误差阈值和 argmax mismatch 通过）。

### Step E：一轮优化（1 天）

只挑一个点做优化并量化：

- 选项1：`row_sum`（shared memory 归约）
- 选项2：`normalize`（访存合并）
- 选项3：kernel 融合（减少 global memory 往返）

验收：至少 1 个 kernel 有可量化提升（时间或吞吐）。

---

## 3. 每一步必须产出的证据

每完成一步都保留以下记录（面试直接可用）：

- 改了什么（1-2 句话）
- 为什么这么改（1 句话）
- 数据结果（正确性或性能）

推荐放到：`rewrite_lab/notes.md`

---

## 4. 最低可投递线（满足就能讲）

满足以下 5 条就可以进简历：

- `rewrite_lab` 可编译运行
- CPU/GPU 正确性对齐
- 至少 1 次 `ncu` 结果
- 至少 1 个优化点有前后对比数据
- 你能在 10 分钟讲清完整优化闭环

---

## 5. 7 天执行版（建议）

- Day1：Step A + Step B（前半）
- Day2：Step B（后半）+ Step C
- Day3：Step D（kernel 1/2）
- Day4：Step D（kernel 3/4 + 联调）
- Day5：正确性修复 + 基准测试
- Day6：Step E 一轮优化 + NCU
- Day7：整理 `notes.md` + 准备简历话术

---

## 6. 实施原则（强约束）

- 每次只动一个变量（方便定位收益）
- 每天至少一次 commit
- 无 benchmark/NCU 数据的“优化”不算优化
- 卡住超过 40 分钟就回退到上一个可运行版本

