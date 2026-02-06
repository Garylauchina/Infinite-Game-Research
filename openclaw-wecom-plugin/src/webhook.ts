/**
 * WeCom Webhook Handler
 *
 * Handles incoming HTTP requests from WeCom:
 * - GET: URL verification (echostr decryption)
 * - POST: Message/event callbacks (decrypt + dispatch to OpenClaw)
 */

import { parseCallbackXml, parseMessageXml } from "./xml.js";
import {
  getCryptoInstance,
  getRuntime,
  getLogger,
  getAllAccountIds,
} from "./runtime.js";
import type { WeComDecryptedMessage } from "./types.js";

/**
 * Create the HTTP handler function for WeCom webhooks
 * This is registered with OpenClaw's gateway HTTP handler system
 */
export function createWeComHttpHandler() {
  const logger = getLogger();

  return async (req: any, res: any) => {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const pathname = url.pathname;

    // Extract account ID from URL path: /webhooks/wecom/:accountId
    const pathParts = pathname.split("/").filter(Boolean);
    let accountId = "default";
    if (pathParts.length >= 3 && pathParts[0] === "webhooks" && pathParts[1] === "wecom") {
      accountId = pathParts[2];
    }

    const cryptoInstance = getCryptoInstance(accountId);
    if (!cryptoInstance) {
      logger.error?.(`[wecom] No crypto instance for account: ${accountId}`);
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Account not found");
      return;
    }

    const method = req.method?.toUpperCase();

    if (method === "GET") {
      await handleVerification(req, res, url, cryptoInstance, logger);
    } else if (method === "POST") {
      await handleCallback(req, res, url, cryptoInstance, accountId, logger);
    } else {
      res.writeHead(405, { "Content-Type": "text/plain" });
      res.end("Method Not Allowed");
    }
  };
}

/**
 * Handle GET request for URL verification
 */
async function handleVerification(
  req: any,
  res: any,
  url: URL,
  crypto: any,
  logger: any
) {
  const msgSignature = url.searchParams.get("msg_signature") || "";
  const timestamp = url.searchParams.get("timestamp") || "";
  const nonce = url.searchParams.get("nonce") || "";
  const echostr = url.searchParams.get("echostr") || "";

  logger.info?.("[wecom] URL verification request received");

  const decrypted = crypto.decryptEchoStr(
    msgSignature,
    timestamp,
    nonce,
    echostr
  );

  if (decrypted !== null) {
    logger.info?.("[wecom] URL verification successful");
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end(decrypted);
  } else {
    logger.error?.("[wecom] URL verification failed - signature mismatch");
    res.writeHead(403, { "Content-Type": "text/plain" });
    res.end("Verification failed");
  }
}

/**
 * Handle POST request for message/event callbacks
 */
async function handleCallback(
  req: any,
  res: any,
  url: URL,
  cryptoInstance: any,
  accountId: string,
  logger: any
) {
  const msgSignature = url.searchParams.get("msg_signature") || "";
  const timestamp = url.searchParams.get("timestamp") || "";
  const nonce = url.searchParams.get("nonce") || "";

  // Read request body
  let body = "";
  for await (const chunk of req) {
    body += chunk;
  }

  logger.info?.("[wecom] Callback received, parsing XML...");

  try {
    // Parse the outer XML to get the encrypted content
    const callbackData = parseCallbackXml(body);
    const encrypt = callbackData.Encrypt;

    if (!encrypt) {
      logger.error?.("[wecom] No Encrypt field in callback XML");
      res.writeHead(400, { "Content-Type": "text/plain" });
      res.end("Bad Request");
      return;
    }

    // Decrypt the message
    const decryptedXml = cryptoInstance.decryptMessage(
      msgSignature,
      timestamp,
      nonce,
      encrypt
    );

    if (decryptedXml === null) {
      logger.error?.("[wecom] Message decryption or verification failed");
      res.writeHead(403, { "Content-Type": "text/plain" });
      res.end("Decryption failed");
      return;
    }

    // Parse the decrypted message XML
    const message = parseMessageXml(decryptedXml) as unknown as WeComDecryptedMessage;

    logger.info?.(
      `[wecom] Message from ${message.FromUserName}: type=${message.MsgType}`
    );

    // Respond immediately to WeCom (within 5 seconds)
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("success");

    // Process the message asynchronously
    await processInboundMessage(message, accountId, logger);
  } catch (error) {
    logger.error?.(`[wecom] Callback processing error: ${error}`);
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Internal Server Error");
  }
}

/**
 * Process an inbound message and dispatch it to the OpenClaw agent
 */
async function processInboundMessage(
  message: WeComDecryptedMessage,
  accountId: string,
  logger: any
) {
  const runtime = getRuntime();

  if (!runtime) {
    logger.error?.("[wecom] Runtime not available, cannot dispatch message");
    return;
  }

  const msgType = message.MsgType;
  let textContent = "";

  switch (msgType) {
    case "text":
      textContent = message.Content || "";
      break;
    case "image":
      textContent = "[图片消息]";
      break;
    case "voice":
      textContent = "[语音消息]";
      break;
    case "video":
      textContent = "[视频消息]";
      break;
    case "location":
      textContent = "[位置消息]";
      break;
    case "link":
      textContent = "[链接消息]";
      break;
    case "event":
      // Handle events (subscribe, click, etc.)
      logger.info?.(
        `[wecom] Event: ${message.Event}, key: ${message.EventKey}`
      );
      return; // Don't dispatch events as messages for now
    default:
      textContent = `[${msgType}消息]`;
  }

  if (!textContent) {
    logger.info?.("[wecom] Empty message content, skipping");
    return;
  }

  // Dispatch to OpenClaw agent pipeline
  try {
    // Build the inbound message context
    const inboundContext = {
      channel: "wecom",
      accountId,
      senderId: message.FromUserName,
      senderName: message.FromUserName,
      chatId: message.FromUserName, // DM uses userId as chatId
      chatType: "direct",
      messageId: message.MsgId || `${Date.now()}`,
      timestamp: message.CreateTime
        ? parseInt(message.CreateTime, 10) * 1000
        : Date.now(),
      text: textContent,
      target: {
        id: message.FromUserName,
        accountId,
      },
    };

    // Try to dispatch via runtime messaging
    if (runtime.messaging?.handleInbound) {
      await runtime.messaging.handleInbound(inboundContext);
      logger.info?.(
        `[wecom] Message dispatched to agent from ${message.FromUserName}`
      );
    } else if (runtime.dispatchInbound) {
      await runtime.dispatchInbound(inboundContext);
      logger.info?.(
        `[wecom] Message dispatched via dispatchInbound from ${message.FromUserName}`
      );
    } else {
      // Fallback: try to use the gateway RPC to dispatch
      logger.warn?.(
        "[wecom] No direct dispatch method found, trying gateway RPC"
      );
      if (runtime.gateway?.rpc) {
        await runtime.gateway.rpc("chat.inbound", inboundContext);
      }
    }
  } catch (error) {
    logger.error?.(`[wecom] Failed to dispatch inbound message: ${error}`);
  }
}
