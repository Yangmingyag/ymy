# scripts/sim_train.py
import numpy as np
import random
import time
import pickle
import os

# ===================== 1. 模拟仿真环境 =====================
class SupermarketEnv:
    """
    模拟一个极其简化的超市货架环境
    真实比赛中，这里会替换成官方提供的仿真环境接口
    """
    def __init__(self):
        # 货架上的商品位置（简化为一维坐标，真实环境是三维空间）
        self.items = [1.0, 2.5, 4.0, 5.5, 7.0]  # 货架上的 5 个商品位置
        self.cart_position = 9.0  # 购物车位置在货架末端
        
        # 当前状态
        self.reset()
    
    def reset(self):
        """重置环境，开始新一回合"""
        # 随机选一个目标商品（比赛时由系统指定）
        self.target_index = random.randint(0, len(self.items) - 1)
        self.target_pos = self.items[self.target_index]
        
        # 机械臂初始位置（在货架起点）
        self.arm_pos = 0.0
        
        # 夹爪状态：0=张开, 1=闭合
        self.gripper = 0
        
        self.steps = 0
        self.max_steps = 50  # 每回合最多 50 步
        self.done = False
        
        return self._get_observation()
    
    def _get_observation(self):
        """生成当前观测（真实比赛是摄像头图像 + 传感器数据）"""
        # 简化：返回 [机械臂位置, 目标位置, 当前与目标距离]
        return np.array([
            self.arm_pos,
            self.target_pos,
            abs(self.arm_pos - self.target_pos)
        ])
    
    def step(self, action):
        """
        执行动作，返回 (观测, 奖励, 是否结束)
        action: [目标位置偏移, 夹爪指令(0或1)]
        """
        self.steps += 1
        
        # 解析动作
        move_delta = action[0] * 0.2  # 每次移动步长受限制
        gripper_cmd = 1 if action[1] > 0.5 else 0
        
        # 更新机械臂位置（夹在货架范围内）
        self.arm_pos += move_delta
        self.arm_pos = max(0.0, min(9.0, self.arm_pos))
        
        # 更新夹爪
        self.gripper = gripper_cmd
        
        # 计算奖励
        distance = abs(self.arm_pos - self.target_pos)
        reward = 0.0
        
        # 如果机械臂到达目标位置且夹爪闭合 -> 抓取成功
        if distance < 0.2 and self.gripper == 1:
            reward = 10.0
            self.done = True
        
        # 如果超时
        if self.steps >= self.max_steps:
            self.done = True
        
        # 返回新观测
        obs = self._get_observation()
        return obs, reward, self.done

# ===================== 2. 定义策略模型 =====================
class SimplePolicy:
    """
    极其简单的策略：根据距离决定移动方向
    真实比赛会用 ACT/Diffusion Policy
    """
    def __init__(self):
        # 随机初始化权重，只是一个演示
        self.threshold = np.random.rand() * 3.0
    
    def act(self, obs):
        """
        根据观测输出动作
        观测: [arm_pos, target_pos, distance]
        动作: [移动方向, 夹爪指令]
        """
        arm_pos, target_pos, distance = obs
        
        # 简单策略：
        # 1. 如果距离 > 0.2，就向着目标移动
        # 2. 如果距离 <= 0.2，就抓取（闭合夹爪）
        if distance > 0.2:
            move = 1.0 if target_pos > arm_pos else -1.0
            gripper = 0.0  # 保持张开
        else:
            move = 0.0
            gripper = 1.0  # 闭合夹爪
        
        return np.array([move, gripper])

# ===================== 3. 训练循环 =====================
def train_policy(env, policy, episodes=100):
    """训练策略（其实是评估，因为我们的策略是硬编码的）"""
    rewards_list = []
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = policy.act(obs)
            obs, reward, done = env.step(action)
            total_reward += reward
        
        rewards_list.append(total_reward)
        success = total_reward >= 10.0
        
        if episode % 20 == 0:
            print(f"回合 {episode}: 奖励 = {total_reward:.1f}  {'✅ 抓取成功' if success else '❌ 失败'}")
    
    return rewards_list

# ===================== 4. 主程序 =====================
if __name__ == "__main__":
    print("🛒 开始仿真超市取货训练...")
    
    # 创建环境和策略
    env = SupermarketEnv()
    policy = SimplePolicy()
    
    # 训练/评估
    rewards = train_policy(env, policy, episodes=100)
    
    # 计算成功率
    success_rate = sum(1 for r in rewards if r >= 10.0) / len(rewards)
    print(f"\n📊 成功率: {success_rate * 100:.1f}%")
    
    # 保存策略（实际比赛中会保存模型权重）
    os.makedirs("models", exist_ok=True)
    with open("models/supermarket_policy.pkl", "wb") as f:
        pickle.dump(policy, f)
    print("✅ 策略已保存至: models/supermarket_policy.pkl")