# 练习路线（V1 -> V4）

> 每个版本都要做 3 件事：  
> 1) 跑通正确性  
> 2) 记录时间/吞吐  
> 3) 做 NCU 分析并写 3 句结论

---

## V1（当前已实现）

内容：

- `add_bias_relu`、`row_sum`、`normalize`、`row_argmax` 分开 kernel
- shared memory 做 block 归约

你要输出：

- CPU vs GPU 的 max diff、argmax mismatch
- 平均耗时与吞吐

---

## V2：Kernel 融合（练 C++ 代码组织 + CUDA 吞吐）

目标：

- 融合 `normalize + argmax`，减少一次全局读写

建议动作：

- 在 `pipeline_gpu.cu` 新增 fused kernel
- 在 `pipeline.h` 新增 `run_gpu_pipeline_v2` 接口
- 在 `main.cu` 增加 `--mode=v1|v2`

---

## V3：向量化访存（练性能敏感代码）

目标：

- 在满足对齐条件时，尝试 `float4` 加载

建议动作：

- 先判断 `dim % 4 == 0`
- 新增 vectorized kernel
- 对比 vectorized on/off 的收益

---

## V4：Warp-level 归约（练并行细节）

目标：

- 用 warp shuffle 减少 shared memory 归约开销

建议动作：

- 对 `row_sum` 或 `row_argmax` 做 warp-level 优化
- 比较 occupancy 与 kernel time 变化

---

## C++ 语法专项练习（穿插做）

1. 给配置结构体加 builder 风格接口（链式调用）  
2. 用 `enum class` 管理执行模式  
3. 用 `std::optional` 表达可选参数  
4. 把输出结果序列化成 CSV（练 `fstream`）  
5. 用 `std::span`（如果编译器支持）改造函数签名

---

## 面试验收标准

- [ ] 你能画出 pipeline 数据流
- [ ] 你能解释每个 kernel 的职责和瓶颈
- [ ] 你能讲清楚一个优化前后证据链
- [ ] 你能说出项目里用了哪些 C++ 工程化手段
- [ ] 你能把这项目讲成 3 分钟故事

