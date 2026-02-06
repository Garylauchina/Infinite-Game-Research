/**
 * WeCom (企业微信) Plugin Type Definitions
 */

export interface WeComAccountConfig {
  /** 企业微信 CorpID */
  corpId: string;
  /** 应用 AgentId */
  agentId: string;
  /** 应用 Secret */
  secret: string;
  /** 回调 Token (用于签名验证) */
  token: string;
  /** 回调 EncodingAESKey (43位, 用于消息加解密) */
  encodingAESKey: string;
  /** 是否启用 */
  enabled?: boolean;
  /** DM 策略 */
  dmPolicy?: "allow" | "allowlisted" | "deny";
  /** 允许的用户列表 */
  allowFrom?: string[];
}

export interface WeComChannelConfig {
  accounts?: Record<string, WeComAccountConfig>;
}

export interface WeComAccessToken {
  token: string;
  expiresAt: number;
}

/** 企业微信回调 GET 请求参数 (URL验证) */
export interface WeComVerifyParams {
  msg_signature: string;
  timestamp: string;
  nonce: string;
  echostr: string;
}

/** 企业微信回调 POST 请求参数 */
export interface WeComCallbackParams {
  msg_signature: string;
  timestamp: string;
  nonce: string;
}

/** 企业微信回调 POST 请求体 (XML解析后) */
export interface WeComCallbackBody {
  xml: {
    ToUserName: string[];
    AgentID: string[];
    Encrypt: string[];
  };
}

/** 解密后的消息结构 */
export interface WeComDecryptedMessage {
  ToUserName: string;
  FromUserName: string;
  CreateTime: string;
  MsgType: string;
  Content?: string;
  MsgId?: string;
  AgentID?: string;
  Event?: string;
  EventKey?: string;
  PicUrl?: string;
  MediaId?: string;
}

/** 发送消息请求 */
export interface WeComSendTextRequest {
  touser: string;
  msgtype: "text";
  agentid: number;
  text: {
    content: string;
  };
}

/** 发送消息响应 */
export interface WeComSendResponse {
  errcode: number;
  errmsg: string;
  invaliduser?: string;
}

/** Access Token 响应 */
export interface WeComTokenResponse {
  errcode: number;
  errmsg: string;
  access_token: string;
  expires_in: number;
}
