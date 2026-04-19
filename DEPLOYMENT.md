# 德勤 AI 定价平台 - 部署指南

## 🚀 快速部署：Streamlit Community Cloud

### 步骤 1: 访问 Streamlit Cloud

访问：https://streamlit.io/cloud

### 步骤 2: 登录并连接 GitHub

1. 使用 GitHub 账户登录
2. 授权 Streamlit 访问您的仓库

### 步骤 3: 创建新应用

1. 点击 "New app"
2. 选择您的仓库：`cyc03357357-rgb/Deloitte-Pricing-`
3. 选择分支：`main`
4. 选择主文件：`deloitte_pricing_platform.py`
5. 为您的应用命名（可选）
6. 点击 "Deploy!"

### 步骤 4: 完成部署！

几秒钟后，您的应用就会上线！您会获得一个链接，例如：
`https://deloitte-pricing.streamlit.app`

## ⚙️ 高级配置

### 添加 Secrets（DeepSeek API Key）

如果您需要使用 DeepSeek AI 功能：

1. 在 Streamlit Cloud 上进入您的应用设置
2. 找到 "Secrets" 部分
3. 添加：
```toml
DEEPSEEK_API_KEY = "your-api-key-here"
```

### 自定义域名

Streamlit Cloud 允许您添加自定义域名（需要付费）。

## 📦 其他部署选项

### 选项 2: Hugging Face Spaces

1. 访问 https://huggingface.co/spaces
2. 创建新 Space
3. 选择 Streamlit SDK
4. 上传代码或连接 GitHub

### 选项 3: Vercel / Railway

这些平台也支持部署 Python/Streamlit 应用。

### 选项 4: 传统服务器（Docker + Nginx）

适用于企业内部部署。

## 📋 部署前检查清单

- [x] 代码已推送到 GitHub
- [x] `requirements.txt` 文件存在
- [x] 主应用文件是 `deloitte_pricing_platform.py`
- [x] `.gitignore` 文件存在（忽略虚拟环境等）
- [ ] （可选）添加了 Streamlit 配置文件

## 🔧 本地测试部署

确保应用在本地正常运行：

```bash
cd "deloitte pricing"
source .venv/bin/activate
pip install -r requirements.txt
streamlit run deloitte_pricing_platform.py
```

## 📞 遇到问题？

- Streamlit 文档：https://docs.streamlit.io/
- 常见问题：https://docs.streamlit.io/streamlit-community-cloud/get-started
