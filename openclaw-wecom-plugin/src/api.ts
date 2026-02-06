/**
 * WeCom (企业微信) REST API Client
 *
 * Handles:
 * - Access token acquisition and caching (auto-refresh before expiry)
 * - Sending text messages to users
 * - Sending markdown messages
 * - Sending image/file messages
 */

import type {
  WeComAccessToken,
  WeComSendTextRequest,
  WeComSendResponse,
  WeComTokenResponse,
} from "./types.js";

const WECOM_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin";

export class WeComApiClient {
  private corpId: string;
  private secret: string;
  private agentId: number;
  private cachedToken: WeComAccessToken | null = null;
  private logger: any;

  constructor(
    corpId: string,
    secret: string,
    agentId: string | number,
    logger?: any
  ) {
    this.corpId = corpId;
    this.secret = secret;
    this.agentId = typeof agentId === "string" ? parseInt(agentId, 10) : agentId;
    this.logger = logger || console;
  }

  /**
   * Get a valid access token, refreshing if needed
   * Token is cached and refreshed 5 minutes before expiry
   */
  async getAccessToken(): Promise<string> {
    const now = Date.now();

    // Return cached token if still valid (with 5 min buffer)
    if (
      this.cachedToken &&
      this.cachedToken.expiresAt > now + 5 * 60 * 1000
    ) {
      return this.cachedToken.token;
    }

    // Fetch new token
    const url = `${WECOM_API_BASE}/gettoken?corpid=${encodeURIComponent(this.corpId)}&corpsecret=${encodeURIComponent(this.secret)}`;

    try {
      const response = await fetch(url);
      const data = (await response.json()) as WeComTokenResponse;

      if (data.errcode !== 0) {
        throw new Error(
          `WeCom gettoken failed: errcode=${data.errcode}, errmsg=${data.errmsg}`
        );
      }

      this.cachedToken = {
        token: data.access_token,
        expiresAt: now + data.expires_in * 1000,
      };

      this.logger.info?.(
        `[wecom] Access token refreshed, expires in ${data.expires_in}s`
      );

      return this.cachedToken.token;
    } catch (error) {
      this.logger.error?.(`[wecom] Failed to get access token: ${error}`);
      throw error;
    }
  }

  /**
   * Send a text message to a user
   */
  async sendText(toUser: string, content: string): Promise<WeComSendResponse> {
    const accessToken = await this.getAccessToken();
    const url = `${WECOM_API_BASE}/message/send?access_token=${accessToken}`;

    const body: WeComSendTextRequest = {
      touser: toUser,
      msgtype: "text",
      agentid: this.agentId,
      text: {
        content,
      },
    };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = (await response.json()) as WeComSendResponse;

      if (data.errcode !== 0) {
        this.logger.error?.(
          `[wecom] Send message failed: errcode=${data.errcode}, errmsg=${data.errmsg}`
        );
      } else {
        this.logger.info?.(`[wecom] Message sent to ${toUser}`);
      }

      return data;
    } catch (error) {
      this.logger.error?.(`[wecom] Failed to send message: ${error}`);
      throw error;
    }
  }

  /**
   * Send a markdown message to a user
   */
  async sendMarkdown(
    toUser: string,
    content: string
  ): Promise<WeComSendResponse> {
    const accessToken = await this.getAccessToken();
    const url = `${WECOM_API_BASE}/message/send?access_token=${accessToken}`;

    const body = {
      touser: toUser,
      msgtype: "markdown",
      agentid: this.agentId,
      markdown: {
        content,
      },
    };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = (await response.json()) as WeComSendResponse;

      if (data.errcode !== 0) {
        this.logger.error?.(
          `[wecom] Send markdown failed: errcode=${data.errcode}, errmsg=${data.errmsg}`
        );
      }

      return data;
    } catch (error) {
      this.logger.error?.(`[wecom] Failed to send markdown: ${error}`);
      throw error;
    }
  }

  /**
   * Send a text card message
   */
  async sendTextCard(
    toUser: string,
    title: string,
    description: string,
    url: string
  ): Promise<WeComSendResponse> {
    const accessToken = await this.getAccessToken();
    const apiUrl = `${WECOM_API_BASE}/message/send?access_token=${accessToken}`;

    const body = {
      touser: toUser,
      msgtype: "textcard",
      agentid: this.agentId,
      textcard: {
        title,
        description,
        url,
      },
    };

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      return (await response.json()) as WeComSendResponse;
    } catch (error) {
      this.logger.error?.(`[wecom] Failed to send text card: ${error}`);
      throw error;
    }
  }
}
