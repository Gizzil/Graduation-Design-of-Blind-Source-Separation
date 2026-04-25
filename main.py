
#### 1.3 main.py（一键运行全流程）
import os
import subprocess
from configs.global_config import make_dirs

if __name__ == "__main__":
    # 初始化所有目录
    make_dirs()
    print("="*50)
    print("盲源分离算法研究 - 全流程实验启动")
    print("="*50)

    # 步骤1：生成数据集
    print("\n[步骤1/3] 开始生成数据集...")
    subprocess.run(["python", "generate_data.py"], check=True)
    print("数据集生成完成！")

    # 步骤2：训练深度学习模型
    print("\n[步骤2/3] 开始训练深度学习分离模型...")
    subprocess.run(["python", "train.py"], check=True)
    print("模型训练完成！")

    # 步骤3：模型测试与对比评估
    print("\n[步骤3/3] 开始模型测试与算法对比...")
    subprocess.run(["python", "test.py"], check=True)
    print("测试与评估完成！")

    print("\n"+"="*50)
    print("全流程实验完成！所有结果已保存至results/目录")
    print("="*50)