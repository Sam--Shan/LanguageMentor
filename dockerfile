# 使用官方Python 3.10精简镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件以利用Docker缓存机制
COPY requirements.txt .

# 安装项目依赖（已添加防止缓存残留的优化参数）
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目文件到容器
COPY . .

# 设置环境变量（如果开发环境有需要可取消注释）
# ENV PYTHONPATH="${PYTHONPATH}:/app/src"
# ENV OPENAI_API_KEY="your-default-key-here"

# 指定容器启动命令
CMD ["python", "src/main.py"]