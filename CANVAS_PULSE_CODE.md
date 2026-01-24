# Canvas 渲染代码 - 脉冲相关部分

## 脉冲数据结构

```javascript
// 脉冲数组
const pulses = [];

// 生成脉冲函数
function spawnPulse(x, y, strength) {
  pulses.push({ x, y, r: 0, s: strength, life: 0 });
  if (pulses.length > 24) pulses.shift();  // 限制最多24个脉冲
}
```

## 脉冲生成逻辑（在 frame 函数中）

```javascript
// 脉冲生成概率：基于波动率 (tremor)
const pulseProb = (0.03 + 0.18 * p.tremor) * dt;

if (Math.random() < pulseProb) {
  // 位置：中心区域 + 随机偏移 + 流动偏移
  const x = innerWidth * (0.45 + 0.10 * (Math.random()-0.5) + 0.10 * p.flow);
  const y = innerHeight * (0.50 + 0.18 * (Math.random()-0.5));
  
  // 强度：基础0.6 + 能量相关0.8
  spawnPulse(x, y, 0.6 + 0.8 * p.energy);
}
```

## 脉冲渲染（在像素循环中）

```javascript
// 计算脉冲对当前像素的影响
let pulseAdd = 0;
for (const pu of pulses) {
  const dx = x - pu.x;
  const dy = y - pu.y;
  const d = Math.sqrt(dx*dx + dy*dy);
  
  // 环形波：exp(-|距离-半径| / 衰减系数)
  const wave = Math.exp(-Math.abs(d - pu.r) / (18 + 22*pu.s));
  pulseAdd += wave * 0.18 * pu.s;
}

// 最终亮度 = 基础亮度 * 发光度 + 脉冲叠加
let br = clamp01(a * glow + pulseAdd);
```

## 脉冲更新（每帧）

```javascript
// 更新所有脉冲
for (const pu of pulses) {
  pu.life += dt;                    // 生命周期增加
  pu.r += dt * (120 + 220*pu.s);   // 半径扩散速度（基于强度）
}

// 移除过期脉冲（生命周期 > 2.2秒）
for (let i=pulses.length-1; i>=0; i--) {
  if (pulses[i].life > 2.2) pulses.splice(i,1);
}
```

## 完整脉冲相关代码片段

```javascript
// ========== 脉冲数据结构 ==========
const pulses = [];
function spawnPulse(x, y, strength) {
  pulses.push({ x, y, r: 0, s: strength, life: 0 });
  if (pulses.length > 24) pulses.shift();
}

// ========== 在 frame() 函数中 ==========

// 1. 生成新脉冲（基于波动率）
const pulseProb = (0.03 + 0.18 * p.tremor) * dt;
if (Math.random() < pulseProb) {
  const x = innerWidth * (0.45 + 0.10 * (Math.random()-0.5) + 0.10 * p.flow);
  const y = innerHeight * (0.50 + 0.18 * (Math.random()-0.5));
  spawnPulse(x, y, 0.6 + 0.8 * p.energy);
}

// 2. 在像素渲染循环中计算脉冲影响
let pulseAdd = 0;
for (const pu of pulses) {
  const dx = x - pu.x;
  const dy = y - pu.y;
  const d = Math.sqrt(dx*dx + dy*dy);
  const wave = Math.exp(-Math.abs(d - pu.r) / (18 + 22*pu.s));
  pulseAdd += wave * 0.18 * pu.s;
}

// 3. 应用到最终亮度
let br = clamp01(a * glow + pulseAdd);

// 4. 更新脉冲状态（帧末）
for (const pu of pulses) {
  pu.life += dt;
  pu.r += dt * (120 + 220*pu.s);
}
for (let i=pulses.length-1; i>=0; i--) {
  if (pulses[i].life > 2.2) pulses.splice(i,1);
}
```

## 关键参数说明

- **pulseProb**: 脉冲生成概率 = `(0.03 + 0.18 * tremor) * dt`
  - 基础概率：3% per frame
  - 波动率影响：最高 +18%
  
- **脉冲强度**: `0.6 + 0.8 * energy`
  - 基础强度：0.6
  - 能量影响：最高 +0.8
  
- **波衰减**: `exp(-|距离-半径| / (18 + 22*强度))`
  - 基础衰减：18
  - 强度影响：最高 +22
  
- **扩散速度**: `120 + 220*强度` 像素/秒
  - 基础速度：120
  - 强度影响：最高 +220
  
- **生命周期**: 2.2秒后自动移除
