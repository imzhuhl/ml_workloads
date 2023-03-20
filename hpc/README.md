# High Performance Compute

## 设备

Linux 平台 Arm Neoverse N2 CPU，后面都基于这个芯片开始实验。

```bash
# 查看 CPU 信息
lscpu
cat /proc/cpuinfo

# 以下是当前实验平台
Architecture:           aarch64
Byte Order:             Little Endian
CPU:                    32
Vendor ID:              ARM
Thread(s) per core:     1
Core(s) per socket:     32
NUMA node(s):           1
CPU MHz:                2750.0000   # 频率
L1d cache:              64K
L1i cache:              64K
L2 cache:               1024K
L3 cache:               65536K


CPU implementer:        0x41        # Arm 公司
CPU part:               0xd49       # Neoverse N2
Features:               fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh
```
Features 列出了当前芯片支持的指令集，可以注意到包含 asimd、sve 等向量指令集。


## GFLOPS 理论值

```
GFLOPS = cores * (cycles / second) * (FLOPs / cycle)
```
GFLOPS 表示每秒浮点数计算次数，G 就是 giga（10的9次方）。FLOPs（s 小写）表示浮点数计算次数。`cycles / second` 就是 CPU 频率。

以 fmla 指令为例：
```c
fmla v0.4s, v1.4s, v2.4s
// v0 = v1 * v2 + v0
```
每个向量寄存器放入四个单精度浮点数，因此 fmla 总共涉及 8 个浮点数操作，4 次乘法和 4 次加法。那么单核 GFLOPS 为：
```
(2750 * 10 ^ 6) * 8 * 2 = 44 GFLOPS
```
最后的 2 是因为每个核中用于处理浮点数计算的 EU (execution unit) 有两个，可以同时执行两句 fmla 指令。


## FMLA 实际 GFLOPS 测量

```c++
void fmla_kernel(int cnt) {
    __asm__ __volatile__(
        "mov x0, %[cnt]\n"
        "1:\n"
        "fmla v0.4s, v1.4s, v2.4s\n"
        "fmla v3.4s, v4.4s, v5.4s\n"
        "fmla v6.4s, v7.4s, v8.4s\n"
        "fmla v9.4s, v10.4s, v11.4s\n"
        "subs x0, x0, #1\n"
        "bne 1b\n"
        :
        : [cnt] "r" (cnt)
        : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5",
          "v6", "v7", "v8", "v9", "v10", "v11"
    );
}
```
循环执行这个函数，然后根据 FLOPs 总数和耗时算出 GFLOPS，详细看代码文件。
实际跑出来的结果是 43.5 GFLOPS 左右，非常接近理论值 44 GFLOPS，符合预期。

这里的实现有一些细节需要分析，例如为什么要在一个循环中放入四个 fmla 指令，如果只放一个会怎么样。
下面手动测试，汇编的循环中只放一个 fmla：
```c++
"mov x0, %[cnt]\n"
"1:\n"
"fmla v0.4s, v1.4s, v2.4s\n"
"subs x0, x0, #1\n"
"bne 1b\n"
```
和只放两个 fmla：
```c++
"mov x0, %[cnt]\n"
"1:\n"
"fmla v0.4s, v1.4s, v2.4s\n"
"fmla v3.4s, v4.4s, v5.4s\n"
"subs x0, x0, #1\n"
"bne 1b\n"
```
会观察到，一个 fmla 时，GFLOPS 不到 11，两个 fmla 时，不到 22 GFLOPS。理论值 44 恰好是这两个值的倍数。

查看 Arm Neoverse N2 的指令手册，有两个信息，fmla 的延迟（latency）是 4，吞吐量（throughput）是 2。吞吐量是 2 表示每个周期（cycle）有两个 fmla 指令完成，也意味着每个周期发射（issue）两个 fmla 指令，由于有两个 EU，可以理解每个 EU 的吞吐量是 1。

按这个理解，可以想象出一个执行 fmla 的流水线场景：
```
x x x x                 // fmla
  x x x x
    x x x x
      x x x x
```
每一个 fmla 的延迟是 4 个周期，而吞吐量是 1。

现在回头看上面汇编的循环中只放一个 fmla 的情况，首先考虑到当前迭代的 fmla 和下一次迭代的 fmla，本来这两者可以间隔一个周期，但是考虑到他们使用了相同的寄存器 v0，并且涉及到对 v0 的写操作。于是流水线暂停了：
```
x x x x             第 n   次迭代的 fmla, v0 = v1 * v2 + v0
        x x x x     第 n+1 次迭代的 fmla, v0 = v1 * v2 + v0
```
指令执行前要准备好所有的操作数，而第 n+1 次迭代的 fmla 使用了 v0，他必须等待第 n 次 fmla 完成，才能开始执行。这种情况下，吞吐量变成了 1/4，而且，2 个 EU 的并行能力也无法发挥，毕竟每条指令都要等待前面的执行完。
这似乎意味 GFLOPS 会是理论峰值的 1/8，即 `(2750 * 10 ^ 6) * 2 * 1 = 5.5 GFLOPS`，但我们实测约 11 GFLOPS，实际快了两倍，这是怎么回事？

仔细看 fmla 的延迟，写着 4(2)，括号中的 2 在页尾解释：
> ASIMD multiply-accumulate pipelines support late-forwarding of accumulate operands from similar μOPs, allowing a typical sequence of floating-point multiply-accumulate μOPs to issue one every N cycles (accumulate latency N shown in parentheses).

简单来说，在执行 fmla 时，是先算乘法，再算加法，那就没必要再一开始就要求三个操作数都准备好，只需先准备好乘法操作数，然后再乘法计算完之前准备好加法操作数就可以了，于是这样 fmla 可以每隔两个周期发射一次：
```
x x x x         第 n   次迭代的 fmla, v0 = v1 * v2 + v0
    x x x x     第 n+1 次迭代的 fmla, v0 = v1 * v2 + v0
```
此时吞吐量是 1/2，比之前提高了一倍，恰好为 11 GFLPS。

在汇编的循环中放两个 fmla，情况类似，由于是两个不相关的 fmla，因此可以利用上双 EU，所以性能翻倍，到了 22 GFLOPS。

而在汇编的循环中放 4 个 fmla 时，恰好每个周期每个 EU 都可以执行完一个 fmla，因此达到了峰值 44 GFLOPS。

late-forwarding 技术使得我们在一个循环中放4个没有依赖的 fmla，就可以打满流水线。否则，我们需要放8个才可以打满流水线。

