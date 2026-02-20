# CUDA + C++ Interview Lab

这个项目专门为你现在的目标设计：

- 做的过程中能反复用到现代 C++ 语法
- 同时练 CUDA kernel 与性能分析
- 最后可以作为 `AI Infra` 实习面试亮点来讲

如果你准备“看懂后从零重写”，请直接看：

- [`README_REWRITE.md`](./README_REWRITE.md)

---

## 这个项目在练什么

实现一个小型推理前处理 Pipeline（按行）：

1. `add_bias + ReLU`
2. `row_sum` 归约
3. `normalize`
4. `row_argmax`

输入是 `batch x dim` 的 `logits`，输出是：

- `normalized`（归一化结果）
- `argmax`（每行最大值下标）
- `max_values`（每行最大值）

---

## 你会练到的 C++ 能力

- `RAII`：`DeviceBuffer<T>` 自动管理 GPU 内存
- 模板：泛型设备缓冲区
- 移动语义：`DeviceBuffer` 支持 move，禁拷贝
- 标准库：`std::vector`、`std::mt19937`、`std::uniform_real_distribution`
- 工程组织：`include/` + `src/` 分层
- 命令行解析：可配置 `batch/dim/warmup/iters/seed`

---

## 你会练到的 CUDA 能力

- kernel 拆分与数据流设计
- shared memory block-level reduction
- grid/block 配置与访存模式
- `cudaEvent` 计时和吞吐统计
- CPU/GPU 正确性对比

---

## 目录结构

```text
cuda_cpp_interview_lab/
├── README.md
├── EXERCISES.md
├── Makefile
├── include/
│   ├── cuda_utils.h
│   ├── pipeline.h
│   └── timer.h
└── src/
    ├── main.cu
    ├── pipeline_cpu.cpp
    └── pipeline_gpu.cu
```

---

## 快速开始

```bash
cd /home/gjj/projects/cuda_cpp_interview_lab
make
./cuda_cpp_interview_lab
```

自定义参数：

```bash
./cuda_cpp_interview_lab --batch=4096 --dim=1024 --warmup=3 --iters=20 --seed=42
```

---

## 面向 RTX 5070 Ti 的编译建议

先看本机 CUDA 工具链版本：

```bash
nvcc --version
```

默认构建使用：

- `CUDA_ARCH=89`
- 同时生成 `sm_89` 与 `compute_89`（含 PTX 兼容）

命令：

```bash
make CUDA_ARCH=89
```

如果你本机 `nvcc` 已支持更高架构（如 Blackwell 对应架构号），可切换：

```bash
make CUDA_ARCH=120
```

---

## NCU 分析

```bash
make profile
```

会生成：

- `reports/pipeline_default.ncu-rep`

---

## 作为面试亮点怎么讲

你可以按这条主线说：

1. 我实现了一个端到端 CUDA 推理前处理 pipeline（而非单个 kernel）
2. 用 C++ RAII + 模板把 GPU 资源管理做了工程化封装
3. 通过 CPU/GPU 对比保证正确性，再用 NCU 观察瓶颈
4. 后续逐步做 kernel 融合、向量化、warp-level 优化

这比“只会写一个 demo kernel”更容易打动 AI Infra 面试官。

---

## 建议你的下一步

先跑通当前版本，然后按 `EXERCISES.md` 从 `V1 -> V4` 迭代，每完成一版都做一次 NCU 报告。
