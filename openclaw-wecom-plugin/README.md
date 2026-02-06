# OpenClaw WeCom (企业微信) Plugin

企业微信 (WeChat Work / WeCom) channel plugin for [OpenClaw](https://openclaw.ai).

## Features

- **Bidirectional Messaging**: Send and receive messages through WeChat Work
- **Message Encryption**: Full WeCom callback encryption/decryption (AES-256-CBC)
- **Signature Verification**: SHA1 signature verification for all callbacks
- **Auto Token Management**: Automatic access_token refresh with caching
- **Long Message Splitting**: Automatically splits messages exceeding WeCom's 2048 char limit
- **Multiple Accounts**: Support for multiple WeCom accounts

## Prerequisites

1. A WeChat Work (企业微信) enterprise account
2. A self-built application (自建应用) with:
   - CorpID (企业ID)
   - AgentId (应用ID)
   - Secret (应用密钥)
3. A callback URL configuration with:
   - Token (回调Token)
   - EncodingAESKey (43位加密密钥)

## Installation

### Local Install (Development)

```bash
openclaw plugins install -l /path/to/openclaw-wecom
```

### Copy Install

```bash
openclaw plugins install /path/to/openclaw-wecom
```

## Configuration

Add the following to your `~/.openclaw/openclaw.json`:

```json
{
  "channels": {
    "wecom": {
      "accounts": {
        "default": {
          "corpId": "wwxxxxxxxxxxxxxxxxx",
          "agentId": "1000002",
          "secret": "your-app-secret-here",
          "token": "your-callback-token",
          "encodingAESKey": "your-43-character-encoding-aes-key",
          "enabled": true
        }
      }
    }
  }
}
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `corpId` | Yes | 企业微信 CorpID (企业ID) |
| `agentId` | Yes | 应用 AgentId |
| `secret` | Yes | 应用 Secret (密钥) |
| `token` | Yes* | 回调 Token (用于签名验证) |
| `encodingAESKey` | Yes* | 回调 EncodingAESKey (43位) |
| `enabled` | No | 是否启用 (默认 true) |

*Required for receiving inbound messages (callback)

## WeCom Callback URL Setup

1. Go to your WeCom admin console (企业微信管理后台)
2. Navigate to your application settings (应用管理 → 自建应用)
3. Find "接收消息" (Receive Messages) section
4. Click "设置API接收" (Set API Receive)
5. Configure:
   - **URL**: `https://your-domain.com/webhooks/wecom/default`
   - **Token**: Same as configured in `token` field
   - **EncodingAESKey**: Same as configured in `encodingAESKey` field
6. Click Save - WeCom will send a verification request to your URL

### URL Format

The webhook URL format is:
```
https://your-domain.com/webhooks/wecom/{accountId}
```

For the default account, use:
```
https://your-domain.com/webhooks/wecom/default
```

## Architecture

```
User (企业微信) ←→ WeCom Server ←→ [Webhook Handler] ←→ OpenClaw Agent
                                         ↕
                                   [WeCom API Client]
                                   (send messages)
```

### Components

- **`src/crypto.ts`** - AES-256-CBC encryption/decryption, SHA1 signature verification
- **`src/api.ts`** - WeCom REST API client (access_token, send messages)
- **`src/webhook.ts`** - HTTP handler for callback verification and message receiving
- **`src/channel.ts`** - OpenClaw channel plugin definition
- **`src/xml.ts`** - XML parsing utilities for WeCom message format
- **`src/runtime.ts`** - Plugin runtime state management
- **`index.ts`** - Plugin entry point and registration

## Troubleshooting

### URL Verification Fails
- Ensure your server is accessible from the internet
- Check that Token and EncodingAESKey match exactly
- Verify your domain has valid HTTPS certificate

### Messages Not Received
- Check gateway logs: `openclaw logs`
- Verify the webhook URL is correct
- Ensure the application has "接收消息" enabled

### Messages Not Sent
- Check access_token is valid (auto-refreshes every 2 hours)
- Verify the user is in the application's visible range (可见范围)
- Check WeCom API error codes in logs

## License

MIT
