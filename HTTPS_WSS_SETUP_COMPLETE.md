# HTTPS/WSS 配置完成

## 实施状态

✅ **已完成**: 无域名方案（自签名证书）

---

## 配置详情

### SSL 证书
- **证书文件**: `/etc/nginx/ssl/infinite-game.crt`
- **私钥文件**: `/etc/nginx/ssl/infinite-game.key`
- **有效期**: 365 天
- **类型**: 自签名证书（CN=45.76.97.37）

### Nginx 配置
- **配置文件**: `/etc/nginx/sites-available/infinite-game`
- **HTTP 端口**: 80（自动重定向到 HTTPS）
- **HTTPS 端口**: 443
- **后端代理**: `http://127.0.0.1:8000`

### WebSocket 支持
- **WSS 端点**: `wss://45.76.97.37/ws/stream`
- **配置**: 已正确设置 `Upgrade` 和 `Connection` 头
- **超时**: 24 小时（86400 秒）

---

## 访问地址

### HTTPS API
```
https://45.76.97.37/health
https://45.76.97.37/incremental?from_t=0&limit=10
https://45.76.97.37/segments
https://45.76.97.37/segment/{filename}
```

### WSS WebSocket
```
wss://45.76.97.37/ws/stream
```

### HTTP（自动重定向）
```
http://45.76.97.37/health  → 自动重定向到 HTTPS
```

---

## 客户端使用

### JavaScript (浏览器)

```javascript
// HTTPS API
fetch('https://45.76.97.37/health')
  .then(r => r.json())
  .then(data => console.log(data));

// WSS WebSocket
const ws = new WebSocket('wss://45.76.97.37/ws/stream');
ws.onmessage = e => {
    const tick = JSON.parse(e.data);
    if (tick.type === 'tick') {
        console.log(`Tick ${tick.t}: ${tick.agents.length} agents`);
    }
};
```

**注意**: 自签名证书会在浏览器显示安全警告，需要点击"高级" → "继续访问"。

### Python

```python
import requests
import asyncio
import websockets
import ssl

# HTTPS API（禁用证书验证，仅用于自签名证书）
response = requests.get('https://45.76.97.37/health', verify=False)
print(response.json())

# WSS WebSocket（禁用证书验证）
ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def connect():
    async with websockets.connect(
        'wss://45.76.97.37/ws/stream',
        ssl=ssl_context
    ) as ws:
        async for message in ws:
            print(message)

asyncio.run(connect())
```

### cURL

```bash
# HTTPS API（跳过证书验证）
curl -k https://45.76.97.37/health

# 或保存证书并验证
curl --cacert /etc/nginx/ssl/infinite-game.crt https://45.76.97.37/health
```

---

## 测试命令

### 测试 HTTPS API
```bash
curl -k https://45.76.97.37/health
curl -k "https://45.76.97.37/incremental?from_t=0&limit=5"
```

### 测试 HTTP 重定向
```bash
curl -I http://45.76.97.37/health
# 应该返回 301 重定向
```

### 测试 WSS（使用 wscat）
```bash
npm install -g wscat
wscat -c wss://45.76.97.37/ws/stream --no-check
```

---

## 服务状态

### 检查 Nginx
```bash
sudo systemctl status nginx
```

### 检查端口
```bash
netstat -tlnp | grep -E ":80|:443"
# 或
ss -tlnp | grep -E ":80|:443"
```

### 查看 Nginx 日志
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 注意事项

### 1. 自签名证书警告
- 浏览器会显示"不安全连接"警告
- 需要手动接受证书（点击"高级" → "继续访问"）
- 生产环境建议使用 Let's Encrypt 证书（需要域名）

### 2. 证书有效期
- 当前证书有效期: 365 天
- 到期前需要重新生成:
  ```bash
  sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/infinite-game.key \
    -out /etc/nginx/ssl/infinite-game.crt \
    -subj "/CN=45.76.97.37/O=Infinite Game/C=CN"
  sudo systemctl reload nginx
  ```

### 3. 防火墙
- 确保 80 和 443 端口开放
- 检查防火墙规则:
  ```bash
  sudo ufw status
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  ```

### 4. 后端服务
- FastAPI 继续监听 8000 端口（仅本地）
- Nginx 作为反向代理，外部访问通过 443 端口

---

## 故障排查

### 问题 1: 无法连接 HTTPS
- 检查 nginx 是否运行: `sudo systemctl status nginx`
- 检查端口是否监听: `netstat -tlnp | grep 443`
- 查看错误日志: `sudo tail -f /var/log/nginx/error.log`

### 问题 2: WebSocket 连接失败
- 检查 nginx 配置中的 `Upgrade` 和 `Connection` 头
- 检查超时设置（应设置为 86400 秒）
- 查看 nginx 访问日志确认请求到达

### 问题 3: 证书错误
- 确认证书文件存在: `ls -lh /etc/nginx/ssl/`
- 检查证书权限: `sudo chmod 644 /etc/nginx/ssl/infinite-game.crt`
- 重新生成证书（见上方）

---

## 升级到 Let's Encrypt（可选）

如果将来有域名，可以升级到 Let's Encrypt 免费证书:

```bash
# 1. 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 2. 配置 DNS: A 记录指向 45.76.97.37

# 3. 获取证书
sudo certbot --nginx -d your-domain.com

# 4. certbot 会自动更新 nginx 配置
```

---

## 总结

✅ **HTTPS/WSS 已启用**
- HTTPS API: `https://45.76.97.37/*`
- WSS WebSocket: `wss://45.76.97.37/ws/stream`
- HTTP 自动重定向到 HTTPS
- 自签名证书（365 天有效期）

⚠️ **注意事项**
- 浏览器会显示证书警告（需要手动接受）
- 证书到期前需要重新生成
- 生产环境建议使用 Let's Encrypt（需要域名）
