# 德勤主题乐园 AI 智能定价平台

## 项目简介

这是一个基于机器学习和 AI 算法的主题乐园智能定价平台，包含多种高级算法和数据分析功能。

## 核心功能

### 1. 方案总览
- 项目介绍和系统架构
- 价格层级对比展示
- 定价策略说明

### 2. 数据看板
- 历史数据可视化
- 客流趋势分析
- 营收数据展示

### 3. AI 定价引擎
- 8种算法协同计算
- 动态参数计算
- 供需关系平衡
- 风险分析评估

### 4. 市场监控
- 未来30天预测
- 风险预警
- DeepSeek AI 智能建议

### 5. GAI 赋能中心
- 定价分析报告生成
- 营销文案生成
- 战略咨询建议

## 核心算法

1. **贝叶斯时间序列预测** - 客流预测
2. **随机森林** - 价格敏感性分析
3. **决策树** - 可解释性分析
4. **强化学习 Q-learning** - 策略优化
5. **梯度提升模型** - 高性能预测
6. **多目标优化** - 营收客流平衡
7. **Monte Carlo 模拟** - 风险分析
8. **模拟退火** - 全局优化

## 安装说明

### 环境要求

- Python 3.8+
- pip 包管理工具

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd deloitte-pricing
```

2. 创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 运行应用
```bash
streamlit run deloitte_pricing_platform.py
```

## 使用说明

### AI 定价引擎

1. 选择运营日期
2. 系统自动获取：日期类型、天气、竞品价格
3. 调整二次消费预期和客流管控目标
4. 点击"启动AI定价计算"
5. 查看定价结果和可视化分析

### GAI 赋能中心

1. 选择日期
2. 系统自动获取相关信息
3. 调整必要参数
4. 生成分析报告或营销文案

## 项目结构

```
deloitte-pricing/
├── deloitte_pricing_platform.py    # 主应用程序
├── README.md                        # 项目说明
├── .gitignore                       # Git 忽略文件
├── .streamlit/                      # Streamlit 配置
└── .venv/                           # 虚拟环境（不提交）
```

## 技术栈

- **Streamlit** - Web 应用框架
- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **Scikit-learn** - 机器学习
- **Plotly** - 可视化
- **OpenAI API** - DeepSeek AI 集成

## 许可证

德勤内部项目

## 联系方式

德勤团队
