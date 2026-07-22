# ===== 练习3：猜数字游戏 =====
import random

print("🎯 猜数字游戏")
print("系统已生成1-100之间的随机数")
print("-" * 30)

secret = random.randint(1, 100)
attempts = 0

while True:
    guess = int(input("请输入你猜的数字："))
    attempts += 1
    
    if guess < secret:
        print("📈 太小了，再大一点！")
    elif guess > secret:
        print("📉 太大了，再小一点！")
    else:
        print(f"🎉 恭喜你！猜对了！")
        print(f"你用了 {attempts} 次猜中")
        break