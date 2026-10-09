# 基本情况
1. 本文件夹记录了通过优化layout以获取比tle算子更好性能的程序和实验记录。每个算子都会有如下文件
- tle算子（从FlagGems或FlagGems-vllm中拷贝而来）
- gluon基线版：用gluon忠实的描述和tle算子一样的算法和layout
- gluon优化版：基于gluon基线版，仅仅改变layout（以及添加或者删减相关的convert layout）
2. 测试环境都是海光bw1000
3. 因为tle算子下降路线上有比gluon更多的优化pass，所以gluon基线版和tle算子的性能也会有差异
4. 加速比的计算方法是 tle算子运行时间 / gluon优化版运行时间

# layout介绍
1. blockedlayout的参数是
```
#ttg.blocked<{
  sizePerThread,
  threadsPerWarp,
  warpsPerCTA,
  order
}>
```