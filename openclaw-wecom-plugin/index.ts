/**
 * OpenClaw WeCom (企业微信) Channel Plugin
 *
 * This plugin enables OpenClaw to communicate through WeChat Work (企业微信),
 * supporting bidirectional messaging:
 * - Outbound: Agent sends messages to WeCom users via REST API
 * - Inbound: Receives user messages via WeCom callback webhook
 *
 * Configuration in openclaw.json:
 * {
 *   "channels": {
 *     "wecom": {
 *       "accounts": {
 *         "default": {
 *           "corpId": "wwxxxxxxxxx",
 *           "agentId": "1000002",
 *           "secret": "your-secret",
 *           "token": "your-callback-token",
 *           "encodingAESKey": "your-43-char-encoding-aes-key",
 *           "enabled": true
 *         }
 *       }
 *     }
 *   }
 * }
 */

import { wecomPlugin } from "./src/channel.js";
import { WeComApiClient } from "./src/api.js";
import { WeComCrypto } from "./src/crypto.js";
import { createWeComHttpHandler } from "./src/webhook.js";
import {
  setRuntime,
  setLogger,
  setApiClient,
  setCryptoInstance,
} from "./src/runtime.js";
import type { WeComAccountConfig } from "./src/types.js";

// Re-export for external use
export { wecomPlugin } from "./src/channel.js";
export { WeComApiClient } from "./src/api.js";
export { WeComCrypto } from "./src/crypto.js";

const plugin = {
  id: "wecom",
  name: "WeCom (企业微信)",
  description:
    "WeChat Work (企业微信) channel plugin for OpenClaw - enterprise messaging integration",

  configSchema: {
    type: "object" as const,
    additionalProperties: false,
    properties: {},
  },

  register(api: any) {
    const logger = api.logger || console;
    setLogger(logger);

    logger.info?.("[wecom] Initializing WeCom plugin...");

    // Store runtime reference for inbound message dispatch
    if (api.runtime) {
      setRuntime(api.runtime);
    }

    // Register the channel
    api.registerChannel({ plugin: wecomPlugin });
    logger.info?.("[wecom] Channel registered");

    // Initialize API clients and crypto instances for each account
    const cfg = api.config || api.runtime?.config;
    if (cfg) {
      initializeAccounts(cfg, logger);
    }

    // Register HTTP handler for WeCom webhooks
    // Route: /webhooks/wecom/:accountId
    if (api.registerHttpHandler) {
      const handler = createWeComHttpHandler();
      api.registerHttpHandler({
        path: "/webhooks/wecom",
        handler,
      });
      logger.info?.(
        "[wecom] HTTP webhook handler registered at /webhooks/wecom/:accountId"
      );
    } else {
      logger.warn?.(
        "[wecom] registerHttpHandler not available - inbound messages will not work"
      );
    }

    // Register a gateway service for periodic token refresh
    if (api.registerService) {
      api.registerService({
        id: "wecom-token-refresh",
        start: () => {
          logger.info?.("[wecom] Token refresh service started");
        },
        stop: () => {
          logger.info?.("[wecom] Token refresh service stopped");
        },
      });
    }

    // Register a status command
    if (api.registerCommand) {
      api.registerCommand({
        name: "wecom-status",
        description: "Show WeCom plugin status",
        handler: () => ({
          text: "WeCom plugin is running. Use /wecom-test to send a test message.",
        }),
      });
    }

    logger.info?.("[wecom] Plugin initialization complete");
  },
};

/**
 * Initialize API clients and crypto instances for all configured accounts
 */
function initializeAccounts(cfg: any, logger: any) {
  const channelCfg = cfg.channels?.wecom;
  if (!channelCfg?.accounts) {
    logger.warn?.("[wecom] No accounts configured under channels.wecom.accounts");
    return;
  }

  for (const [accountId, account] of Object.entries(channelCfg.accounts)) {
    const acct = account as WeComAccountConfig;

    if (acct.enabled === false) {
      logger.info?.(`[wecom] Account ${accountId} is disabled, skipping`);
      continue;
    }

    if (!acct.corpId || !acct.secret || !acct.agentId) {
      logger.error?.(
        `[wecom] Account ${accountId} missing required fields (corpId, secret, agentId)`
      );
      continue;
    }

    // Create API client
    const apiClient = new WeComApiClient(
      acct.corpId,
      acct.secret,
      acct.agentId,
      logger
    );
    setApiClient(accountId, apiClient);

    // Create crypto instance (for callback handling)
    if (acct.token && acct.encodingAESKey) {
      const crypto = new WeComCrypto(
        acct.token,
        acct.encodingAESKey,
        acct.corpId
      );
      setCryptoInstance(accountId, crypto);
      logger.info?.(
        `[wecom] Account ${accountId} initialized with crypto (callback enabled)`
      );
    } else {
      logger.info?.(
        `[wecom] Account ${accountId} initialized (outbound only, no callback token/key)`
      );
    }

    // Pre-fetch access token
    apiClient.getAccessToken().catch((err) => {
      logger.error?.(
        `[wecom] Failed to pre-fetch access token for ${accountId}: ${err}`
      );
    });
  }
}

export default plugin;
