# scripts/sim_train.py
import numpy as np
import time
import pickle
import os

# 模拟生成训练数据
def generate_dummy_data(num_samples=100):
    X = []
    y = []
    for _ in range(num_samples):
        red_intensity = np.random.rand()
        action = 45 if red_intensity > 0.5 else -10
        X.append([red_intensity])
        y.append([action])
    return np.array(X), np.array(y)

# 简单模型
class SimplePolicy:
    def __init__(self):
        self.weights = np.random.randn(1, 1)
        self.bias = np.random.randn(1)
    
    def train(self, X, y, epochs=20):
        lr = 0.01
        for epoch in range(epochs):
            pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((pred - y) ** 2)
            grad_w = np.mean((pred - y) * X, axis=0)
            grad_b = np.mean(pred - y, axis=0)
            self.weights -= lr * grad_w.reshape(self.weights.shape)
            self.bias -= lr * grad_b
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    print("🚀 开始模拟模型训练...")
    X_train, y_train = generate_dummy_data(200)
    policy = SimplePolicy()
    policy.train(X_train, y_train, epochs=50)
    os.makedirs("models", exist_ok=True)
    with open("models/demo_policy.pkl", "wb") as f:
        pickle.dump(policy, f)
    print("✅ 模型已保存至: models/demo_policy.pkl")
    test_input = np.array([[0.8]])
    prediction = policy.predict(test_input)
    print(f"🔮 推理结果: 输入强度 0.8 -> 预测动作角度: {prediction[0][0]:.2f}°")