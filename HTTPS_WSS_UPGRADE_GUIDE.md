# HTTPS/WSS 升级指南

## 概述

将 Infinite Game 服务器从 HTTP/WS 升级到 HTTPS/WSS，提供加密连接支持。

---

## 方案对比

### 方案 1: Nginx 反向代理 + Let's Encrypt（推荐）⭐⭐⭐⭐⭐

**架构**:
```
客户端 → HTTPS (443) → Nginx (SSL终止) → HTTP (8000) → FastAPI
客户端 → WSS (443) → Nginx (WebSocket代理) → WS (8000) → FastAPI
```

**优点**:
- ✅ 实现简单，无需修改 Python 代码
- ✅ Nginx 处理 SSL/TLS，性能好
- ✅ Let's Encrypt 免费证书
- ✅ 支持 HTTP/2
- ✅ 可以配置缓存、限流等

**缺点**:
- ⚠️ 需要配置 nginx
- ⚠️ 需要域名（Let's Encrypt 要求）

**实施复杂度**: 低

---

### 方案 2: FastAPI 直接支持 HTTPS ⭐⭐⭐

**架构**:
```
客户端 → HTTPS (443) → FastAPI (直接处理 SSL)
客户端 → WSS (443) → FastAPI (直接处理 SSL)
```

**优点**:
- ✅ 无需 nginx
- ✅ 配置简单

**缺点**:
- ⚠️ Python 处理 SSL 性能较低
- ⚠️ 需要自己管理证书
- ⚠️ 不支持 HTTP/2

**实施复杂度**: 中

---

### 方案 3: 使用 uvicorn 的 SSL 支持 ⭐⭐⭐

**架构**:
```
客户端 → HTTPS (443) → Uvicorn (SSL) → FastAPI
```

**优点**:
- ✅ 无需 nginx
- ✅ uvicorn 内置 SSL 支持

**缺点**:
- ⚠️ 性能不如 nginx
- ⚠️ 需要自己管理证书

**实施复杂度**: 低

---

## 推荐方案：Nginx 反向代理

### 架构图

```
Internet
   │
   ├─ HTTPS (443) ──→ Nginx ──→ HTTP (8000) ──→ FastAPI
   │                      │
   └─ WSS (443) ────────→ Nginx ──→ WS (8000) ──→ FastAPI
```

### 优势
1. **性能**: Nginx 专门优化 SSL/TLS 处理
2. **灵活性**: 可以配置多个后端、负载均衡
3. **安全性**: 可以添加 WAF、限流等
4. **维护**: 证书更新由 certbot 自动处理

---

## 实施步骤

### 前置条件

1. **域名**: 需要一个域名指向服务器 IP（如 `infinite-game.example.com`）
2. **DNS 配置**: A 记录指向 `45.76.97.37`
3. **防火墙**: 开放 80 和 443 端口

### 步骤 1: 安装 Certbot

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install certbot python3-certbot-nginx -y

# 或使用 snap
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```

### 步骤 2: 配置 Nginx（基础配置）

创建配置文件 `/etc/nginx/sites-available/infinite-game`:

```nginx
# HTTP 服务器（用于 Let's Encrypt 验证和重定向）
server {
    listen 80;
    server_name 45.76.97.37;  # 或使用域名 infinite-game.example.com
    
    # Let's Encrypt 验证
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # 重定向到 HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS 服务器
server {
    listen 443 ssl http2;
    server_name 45.76.97.37;  # 或使用域名
    
    # SSL 证书（将由 certbot 自动配置）
    # ssl_certificate /etc/letsencrypt/live/your-domain/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;
    
    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # API 代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    
    # WebSocket 端点（显式配置）
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 超时
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

### 步骤 3: 启用配置

```bash
# 创建符号链接
sudo ln -s /etc/nginx/sites-available/infinite-game /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载 nginx
sudo systemctl reload nginx
```

### 步骤 4: 获取 SSL 证书

#### 选项 A: 使用域名（推荐）

```bash
# 使用 certbot 自动配置
sudo certbot --nginx -d infinite-game.example.com

# 或手动获取证书
sudo certbot certonly --nginx -d infinite-game.example.com
```

#### 选项 B: 使用 IP 地址（需要自签名证书）

```bash
# 生成自签名证书
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/infinite-game.key \
  -out /etc/nginx/ssl/infinite-game.crt \
  -subj "/CN=45.76.97.37"

# 更新 nginx 配置使用自签名证书
```

**注意**: 自签名证书会在浏览器显示警告，仅适合内部使用。

### 步骤 5: 更新 Nginx 配置（使用证书）

Certbot 会自动更新配置文件，或手动添加：

```nginx
ssl_certificate /etc/letsencrypt/live/infinite-game.example.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/infinite-game.example.com/privkey.pem;
```

### 步骤 6: 配置自动续期

```bash
# 测试续期
sudo certbot renew --dry-run

# 添加到 crontab（自动续期）
sudo crontab -e
# 添加: 0 0 * * * certbot renew --quiet
```

---

## 客户端使用

### HTTPS API

```bash
# 之前: http://45.76.97.37:8000/health
# 现在: https://45.76.97.37/health

curl https://45.76.97.37/health
```

### WSS WebSocket

```javascript
// 之前: ws://45.76.97.37:8000/ws/stream
// 现在: wss://45.76.97.37/ws/stream

const ws = new WebSocket('wss://45.76.97.37/ws/stream');
ws.onmessage = e => console.log(JSON.parse(e.data));
```

### Python 客户端

```python
import asyncio
import websockets
import ssl

# WSS 连接
async def connect():
    # 如果是自签名证书，需要禁用验证（仅开发环境）
    ssl_context = ssl.SSLContext()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    async with websockets.connect(
        'wss://45.76.97.37/ws/stream',
        ssl=ssl_context  # 生产环境应使用有效证书
    ) as ws:
        async for message in ws:
            print(message)

asyncio.run(connect())
```

---

## 备选方案：Uvicorn 直接 SSL

如果不想使用 nginx，可以直接在 uvicorn 配置 SSL：

### 修改 server.py

```python
if __name__ == "__main__":
    init_data_dir()
    
    th = threading.Thread(target=run_forever, daemon=True)
    th.start()
    
    import uvicorn
    
    # SSL 配置
    ssl_keyfile = os.environ.get("SSL_KEYFILE", "/path/to/key.pem")
    ssl_certfile = os.environ.get("SSL_CERTFILE", "/path/to/cert.pem")
    
    port = int(os.environ.get("PORT", "443"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        ssl_keyfile=ssl_keyfile if os.path.exists(ssl_keyfile) else None,
        ssl_certfile=ssl_certfile if os.path.exists(ssl_certfile) else None,
        access_log=False,
        log_level="warning"
    )
```

### 生成自签名证书

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=45.76.97.37"
```

---

## 安全配置建议

### 1. SSL/TLS 配置

```nginx
# 现代 SSL 配置
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
```

### 2. 安全头

```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
```

### 3. 限流（防止滥用）

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location / {
    limit_req zone=api_limit burst=20;
    # ... 其他配置
}
```

---

## 测试检查清单

- [ ] HTTPS 访问正常: `curl https://your-domain/health`
- [ ] WSS 连接正常: 浏览器控制台测试
- [ ] HTTP 自动重定向到 HTTPS
- [ ] 证书自动续期配置
- [ ] 防火墙开放 443 端口
- [ ] 安全头正确设置

---

## 注意事项

1. **域名要求**: Let's Encrypt 需要域名，不能直接用 IP
2. **证书续期**: Let's Encrypt 证书 90 天过期，需要自动续期
3. **防火墙**: 确保 443 端口开放
4. **性能**: Nginx 反向代理对性能影响很小（<5%）
5. **WebSocket 升级**: Nginx 需要正确配置 `Upgrade` 和 `Connection` 头

---

## 快速实施（使用域名）

```bash
# 1. 安装 certbot
sudo apt install certbot python3-certbot-nginx -y

# 2. 配置 DNS: A 记录指向 45.76.97.37

# 3. 创建 nginx 配置（见上方）

# 4. 获取证书
sudo certbot --nginx -d your-domain.com

# 5. 完成！certbot 会自动配置 nginx
```

---

## 快速实施（仅 IP，自签名证书）

```bash
# 1. 创建 SSL 目录
sudo mkdir -p /etc/nginx/ssl

# 2. 生成自签名证书
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/infinite-game.key \
  -out /etc/nginx/ssl/infinite-game.crt \
  -subj "/CN=45.76.97.37"

# 3. 配置 nginx（使用自签名证书路径）

# 4. 重载 nginx
sudo systemctl reload nginx
```

---

## 总结

**推荐方案**: Nginx 反向代理 + Let's Encrypt

**优势**:
- ✅ 性能好
- ✅ 免费证书
- ✅ 自动续期
- ✅ 无需修改 Python 代码

**实施时间**: 约 15-30 分钟

**需要**: 域名（Let's Encrypt）或自签名证书（仅 IP）
