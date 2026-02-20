# 从零重写指南（强烈推荐）

先回答你的问题：**我非常鼓励你“新开空目录重写”**。  
因为面试真正看重的是：

- 你能不能自己从 0 到 1 搭起来
- 你能不能解释每个设计取舍
- 你能不能在出 bug 时独立定位

“看懂”是第一层；“重写出来并讲清楚”才是高含金量。

---

## 重写总原则

1. **禁止复制粘贴原实现**：最多参考接口，不直接抄函数体  
2. **先正确，再性能**：先 CPU/GPU 对齐，再谈优化  
3. **每次只改一件事**：便于定位问题与记录收益  
4. **每完成一阶段都留痕**：记录结论、耗时、问题与修复

---

## 建议目录（在本项目里新开）

建议你在当前项目下新建：

```bash
mkdir -p rewrite_lab/include rewrite_lab/src
```

这样你可以：

- 保留参考实现用于对照
- 在 `rewrite_lab/` 里从零重写
- 后面面试时可以展示“演进过程”

---

## 推荐学习顺序（先看再写）

1. `README.md`：搞清项目目标与整体数据流  
2. `include/pipeline.h`：先理解接口和数据结构  
3. `src/pipeline_cpu.cpp`：理解正确性基准逻辑  
4. `src/main.cu`：理解程序入口、参数、校验输出  
5. `include/cuda_utils.h`：理解 RAII + 模板封装  
6. `src/pipeline_gpu.cu`：最后看 CUDA 内核与 launch

这样顺序的原因是：**先有“契约和答案”，再做 GPU 实现**，调试成本最低。

---

## 从零重写顺序（先写哪些文件，为什么）

### Step 1：先写 `rewrite_lab/include/pipeline.h`

先定义：

- `PipelineConfig`
- `PipelineOutputs`
- `GpuRunStats`
- `run_cpu_pipeline` / `run_gpu_pipeline_v1` 接口

为什么先写它：

- 它是全项目“接口契约”
- 先定边界，后续实现不容易乱

---

### Step 2：写 `rewrite_lab/src/pipeline_cpu.cpp`

先只做 CPU 版本：

- `add_bias + ReLU`
- `row_sum`
- `normalize`
- `argmax`

为什么先写它：

- 它是 GPU 的正确性金标准
- 没有 CPU baseline，GPU 很难判断对错

---

### Step 3：写 `rewrite_lab/src/main.cu`（先只接 CPU）

先完成：

- 参数解析
- 随机输入生成
- 调用 CPU pipeline
- 打印时间与基本信息

为什么这一步很关键：

- 先打通主流程，后面只需“替换后端”
- 能快速确认工程骨架没问题

---

### Step 4：写 `rewrite_lab/include/cuda_utils.h`

实现：

- `CUDA_CHECK`
- `DeviceBuffer<T>`（禁拷贝、可移动、自动释放）

为什么现在写：

- 这一步把“易错资源管理”一次性规范化
- 后面写 kernel 时更专注算法，不被内存管理拖垮

---

### Step 5：写 `rewrite_lab/src/pipeline_gpu.cu`（V1）

按顺序实现 4 个 kernel：

1. `add_bias_relu_kernel`
2. `row_sum_kernel`
3. `normalize_kernel`
4. `row_argmax_kernel`

再写 `launch_pipeline_v1` 和 `run_gpu_pipeline_v1`。

为什么按这个顺序：

- 它对应真实数据流，最容易定位中间错误
- 每个 kernel 都能单独验证

---

### Step 6：回到 `rewrite_lab/src/main.cu` 接入 GPU 对比

加上：

- CPU/GPU `max_abs_diff`
- `argmax mismatch`
- PASS/FAIL 判定

为什么最后接：

- 避免早期把所有复杂度混在一起
- 能保证每步都可测、可解释

---

## 每一步写完要自问的 3 个问题

1. 这一步的输入/输出契约是什么？  
2. 出错时我该看哪个中间量？  
3. 这一步如果优化，最可能影响哪个指标？

---

## 你完成重写后，面试可讲亮点

- 我先用 CPU 建立 correctness baseline，再做 GPU 版本  
- 我用 RAII + 模板封装 GPU 资源管理，减少泄漏和重复代码  
- 我按 kernel 数据流拆解实现，逐步验证而不是一次写完  
- 我可以继续演进到 V2/V3/V4，并用 NCU 量化收益

---

## 通过线（满足就可以拿去讲）

- [ ] `rewrite_lab` 可独立编译运行  
- [ ] CPU/GPU 对齐（误差和 mismatch 在阈值内）  
- [ ] 你能不看代码复述每个文件职责  
- [ ] 你能解释为什么先写这些文件而不是直接写 kernel  
- [ ] 你能给出下一步优化计划（V2/V3/V4）

