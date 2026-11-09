
# -*- coding: utf-8 -*-
"""小批量梯度下降里的大数定律——batch size 如何控制梯度估计的方差

核心实验：
1. 在手写数字数据集（8x8=64维，10类）上训练多分类逻辑回归；
2. 对比不同 batch size（1, 8, 32, 128, full batch）下：
   - 梯度估计的实际方差（理论值 σ²/B）
   - 训练损失收敛速度
   - 测试集准确率
3. 验证大数定律：batch 越大，梯度估计越准（方差小），但泛化未必更好。
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(X, W, b):
    return softmax(X @ W + b)


def loss_and_grad(X, y, W, b):
    n = X.shape[0]
    p = predict_proba(X, W, b)
    # 交叉熵损失
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    # 梯度
    d = p.copy()
    d[np.arange(n), y] -= 1
    d /= n
    grad_W = X.T @ d
    grad_b = d.mean(axis=0)
    return loss, grad_W, grad_b


def train_sgd(X_train, y_train, X_test, y_test, batch_size, lr=0.1, epochs=30, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    c = 10
    W = rng.normal(0, 0.01, size=(d, c))
    b = np.zeros(c)

    history = {'loss': [], 'test_acc': [], 'grad_var_trace': []}

    for epoch in range(epochs):
        # 每个 epoch 打乱
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        # 记录每个 batch 梯度估计的方差（采样10个batch）
        grad_samples = []
        for i in range(0, n, batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]
            _, gW, gb = loss_and_grad(xb, yb, W, b)
            grad_samples.append(gW.flatten())
            # 一步更新
            W -= lr * gW
            b -= lr * gb

        # 记录整个 epoch 结束后的损失和准确率
        train_loss, _, _ = loss_and_grad(X_train, y_train, W, b)
        test_acc = (predict_proba(X_test, W, b).argmax(axis=1) == y_test).mean()
        history['loss'].append(train_loss)
        history['test_acc'].append(test_acc)

        # 计算梯度估计方差（对这一epoch内的batch梯度做样本方差）
        grad_stack = np.array(grad_samples)
        grad_var = grad_stack.var(axis=0).mean()  # 所有维度的平均方差
        history['grad_var_trace'].append(grad_var)

    return history


# 加载数据
digits = load_digits()
X = digits.data
y = digits.target
# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本, 类别: 10')
print()

# 对比不同 batch size
batch_sizes = [1, 8, 32, 128, X_train.shape[0]]
results = {}

for B in batch_sizes:
    B_label = f'{B}' if B < X_train.shape[0] else 'full'
    hist = train_sgd(X_train, y_train, X_test, y_test, batch_size=B, lr=0.1, epochs=30)
    results[B_label] = hist
    final_acc = hist['test_acc'][-1]
    final_loss = hist['loss'][-1]
    final_grad_var = hist['grad_var_trace'][-1]
    # 理论方差 = σ²/B，这里 σ² 用 full batch 梯度方差近似
    print(f'batch_size={B_label:>5}:  最终测试准确率={final_acc:.4f},  最终训练损失={final_loss:.4f},  梯度估计方差={final_grad_var:.6f}')

print()
print('=== 理论验证：梯度估计方差 ∝ 1/B ===')
# 用 batch=1 时的梯度方差作为 σ² 近似，验证 σ²/B 的关系
var_b1 = results['1']['grad_var_trace'][-1]
print(f'以 batch=1 的梯度方差为 σ² ≈ {var_b1:.6f}')
for B_label in ['8', '32', '128', 'full']:
    B_val = int(B_label) if B_label != 'full' else X_train.shape[0]
    theoretical = var_b1 / B_val
    actual = results[B_label]['grad_var_trace'][-1]
    print(f'  B={B_label:>5}: 理论 σ²/B={theoretical:.6f}, 实际={actual:.6f}, 比值={actual/theoretical:.2f}')
