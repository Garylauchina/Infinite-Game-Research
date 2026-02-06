/**
 * WeCom Channel Plugin Definition
 *
 * Defines the channel object that OpenClaw uses to:
 * - Identify and configure the WeCom channel
 * - Route outbound messages to WeCom users
 * - Handle account resolution
 */

import type { WeComAccountConfig, WeComChannelConfig } from "./types.js";
import { getApiClient, getLogger } from "./runtime.js";

/**
 * Split long text into chunks that fit WeCom's 2048 char limit
 */
function splitMessage(text: string, maxLen = 2000): string[] {
  if (text.length <= maxLen) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    if (remaining.length <= maxLen) {
      chunks.push(remaining);
      break;
    }

    // Try to split at a newline
    let splitAt = remaining.lastIndexOf("\n", maxLen);
    if (splitAt < maxLen * 0.5) {
      // If no good newline, split at space
      splitAt = remaining.lastIndexOf(" ", maxLen);
    }
    if (splitAt < maxLen * 0.3) {
      // If no good space, hard split
      splitAt = maxLen;
    }

    chunks.push(remaining.substring(0, splitAt));
    remaining = remaining.substring(splitAt).trimStart();
  }

  return chunks;
}

export const wecomPlugin = {
  id: "wecom",

  meta: {
    id: "wecom",
    label: "WeCom",
    selectionLabel: "WeCom / 企业微信 (WeChat Work)",
    docsPath: "/channels/wecom",
    docsLabel: "wecom",
    blurb: "企业微信 (WeChat Work) enterprise messaging integration.",
    aliases: ["wechat-work", "wework"],
  },

  capabilities: {
    chatTypes: ["direct"] as const,
  },

  config: {
    listAccountIds: (cfg: any): string[] => {
      const channelCfg = cfg.channels?.wecom as WeComChannelConfig | undefined;
      return Object.keys(channelCfg?.accounts ?? {});
    },

    resolveAccount: (cfg: any, accountId?: string): WeComAccountConfig => {
      const channelCfg = cfg.channels?.wecom as WeComChannelConfig | undefined;
      const id = accountId ?? "default";
      return (
        channelCfg?.accounts?.[id] ?? ({ accountId: id } as any)
      );
    },
  },

  outbound: {
    deliveryMode: "direct" as const,

    /**
     * Send a text message through WeCom
     * Called by OpenClaw when the agent needs to reply
     */
    sendText: async (params: {
      text: string;
      target?: { id?: string; accountId?: string };
      accountId?: string;
    }): Promise<{ ok: boolean }> => {
      const logger = getLogger();
      const accountId = params.accountId ?? params.target?.accountId ?? "default";
      const userId = params.target?.id;

      if (!userId) {
        logger.error?.("[wecom] No target user ID for sendText");
        return { ok: false };
      }

      const client = getApiClient(accountId);
      if (!client) {
        logger.error?.(
          `[wecom] No API client for account ${accountId}`
        );
        return { ok: false };
      }

      try {
        // Split long messages
        const chunks = splitMessage(params.text);

        for (const chunk of chunks) {
          const result = await client.sendText(userId, chunk);
          if (result.errcode !== 0) {
            logger.error?.(
              `[wecom] Send failed: ${result.errcode} ${result.errmsg}`
            );
            return { ok: false };
          }
        }

        return { ok: true };
      } catch (error) {
        logger.error?.(`[wecom] sendText error: ${error}`);
        return { ok: false };
      }
    },
  },
};
