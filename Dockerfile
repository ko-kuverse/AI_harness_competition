# 智训引擎 Harness 运行镜像。只装 numpy 与 swanlab，不含 GPU 依赖，不含大模型权重。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_CPU_THREADS=1 \
    HARNESS_SWANLAB_MODE=offline

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ harness/
COPY configs/ configs/
# 官方基线包里的训练代码与静态基线检查点。平台另行挂载基线包时，用 --baseline-dir 指向挂载位置。
COPY cpu-c-guide-20261110-v1/baselines/ baseline/baselines/
COPY cpu-c-guide-20261110-v1/checkpoint.npz cpu-c-guide-20261110-v1/training.json baseline/

# /data/stream 放批次流（每个子目录一批），/work 放状态、注册表、决策与实验记录
VOLUME ["/data", "/work"]
ENTRYPOINT ["python", "-m", "harness"]
CMD ["run", "--config", "configs/harness.json", "--baseline-dir", "baseline", "--stream", "/data/stream", "--workdir", "/work"]
