/**
 * Test script for WeCom crypto module
 * Run with: node test_crypto.mjs
 */

import crypto from "node:crypto";

// ============================================================
// Inline the WeComCrypto class for testing (since we can't
// import .ts directly in plain Node without jiti)
// ============================================================

class WeComCrypto {
  constructor(token, encodingAESKey, corpId) {
    this.token = token;
    this.encodingAESKey = encodingAESKey;
    this.corpId = corpId;
    this.aesKey = Buffer.from(encodingAESKey + "=", "base64");
    this.iv = this.aesKey.subarray(0, 16);
  }

  computeSignature(timestamp, nonce, encrypt) {
    const parts = [this.token, timestamp, nonce, encrypt].sort();
    const str = parts.join("");
    return crypto.createHash("sha1").update(str).digest("hex");
  }

  verifySignature(msgSignature, timestamp, nonce, encrypt) {
    const computed = this.computeSignature(timestamp, nonce, encrypt);
    return computed === msgSignature;
  }

  encrypt(message) {
    const random = crypto.randomBytes(16);
    const msgBuffer = Buffer.from(message, "utf-8");
    const msgLen = Buffer.alloc(4);
    msgLen.writeUInt32BE(msgBuffer.length, 0);
    const corpIdBuffer = Buffer.from(this.corpId, "utf-8");

    let plaintext = Buffer.concat([random, msgLen, msgBuffer, corpIdBuffer]);

    const blockSize = 32;
    const padLen = blockSize - (plaintext.length % blockSize);
    const padding = Buffer.alloc(padLen, padLen);
    plaintext = Buffer.concat([plaintext, padding]);

    const cipher = crypto.createCipheriv("aes-256-cbc", this.aesKey, this.iv);
    cipher.setAutoPadding(false);

    const encrypted = Buffer.concat([cipher.update(plaintext), cipher.final()]);
    return encrypted.toString("base64");
  }

  decrypt(encrypted) {
    const encryptedBuffer = Buffer.from(encrypted, "base64");

    const decipher = crypto.createDecipheriv("aes-256-cbc", this.aesKey, this.iv);
    decipher.setAutoPadding(false);

    let decrypted = Buffer.concat([decipher.update(encryptedBuffer), decipher.final()]);

    const padLen = decrypted[decrypted.length - 1];
    decrypted = decrypted.subarray(0, decrypted.length - padLen);

    const msgLen = decrypted.readUInt32BE(16);
    const message = decrypted.subarray(20, 20 + msgLen).toString("utf-8");
    const corpId = decrypted.subarray(20 + msgLen).toString("utf-8");

    return { message, corpId };
  }

  decryptEchoStr(msgSignature, timestamp, nonce, echostr) {
    if (!this.verifySignature(msgSignature, timestamp, nonce, echostr)) {
      return null;
    }
    const { message } = this.decrypt(echostr);
    return message;
  }

  decryptMessage(msgSignature, timestamp, nonce, encrypt) {
    if (!this.verifySignature(msgSignature, timestamp, nonce, encrypt)) {
      return null;
    }
    const { message, corpId } = this.decrypt(encrypt);
    if (corpId !== this.corpId) {
      return null;
    }
    return message;
  }

  generateReplyXml(replyMsg, timestamp, nonce) {
    const encrypt = this.encrypt(replyMsg);
    const signature = this.computeSignature(timestamp, nonce, encrypt);
    return `<xml>
<Encrypt><![CDATA[${encrypt}]]></Encrypt>
<MsgSignature><![CDATA[${signature}]]></MsgSignature>
<TimeStamp>${timestamp}</TimeStamp>
<Nonce><![CDATA[${nonce}]]></Nonce>
</xml>`;
  }
}

// ============================================================
// XML parsing functions
// ============================================================

function parseCallbackXml(xml) {
  return {
    ToUserName: extractCData(xml, "ToUserName"),
    AgentID: extractCData(xml, "AgentID"),
    Encrypt: extractCData(xml, "Encrypt"),
  };
}

function parseMessageXml(xml) {
  const result = {};
  const cdataRegex = /<(\w+)><!\[CDATA\[(.*?)\]\]><\/\1>/gs;
  const plainRegex = /<(\w+)>([^<]+)<\/\1>/g;
  let match;
  while ((match = cdataRegex.exec(xml)) !== null) {
    result[match[1]] = match[2];
  }
  while ((match = plainRegex.exec(xml)) !== null) {
    if (!(match[1] in result)) {
      result[match[1]] = match[2];
    }
  }
  return result;
}

function extractCData(xml, tagName) {
  const cdataRegex = new RegExp(
    `<${tagName}><!\\[CDATA\\[([\\s\\S]*?)\\]\\]></${tagName}>`,
    "i"
  );
  const cdataMatch = xml.match(cdataRegex);
  if (cdataMatch) return cdataMatch[1];
  const plainRegex = new RegExp(`<${tagName}>([^<]*)</${tagName}>`, "i");
  const plainMatch = xml.match(plainRegex);
  if (plainMatch) return plainMatch[1];
  return "";
}

// ============================================================
// Tests
// ============================================================

console.log("=== WeCom Crypto Module Tests ===\n");

// Test parameters (using the user's actual credentials)
const TOKEN = "testtoken123";
const ENCODING_AES_KEY = "7jlP5lfbevs1RsgfO5jhTRaNvMFNe4xF3yvz567TxN1";
const CORP_ID = "wwc6c7670ccb97416b";

const wecomCrypto = new WeComCrypto(TOKEN, ENCODING_AES_KEY, CORP_ID);

// Test 1: Encrypt and Decrypt roundtrip
console.log("Test 1: Encrypt/Decrypt roundtrip");
const testMessage = '<xml><ToUserName><![CDATA[wwc6c7670ccb97416b]]></ToUserName><FromUserName><![CDATA[testuser]]></FromUserName><CreateTime>1348831860</CreateTime><MsgType><![CDATA[text]]></MsgType><Content><![CDATA[Hello OpenClaw!]]></Content><MsgId>1234567890123456</MsgId><AgentID>1000002</AgentID></xml>';

const encrypted = wecomCrypto.encrypt(testMessage);
console.log(`  Encrypted length: ${encrypted.length}`);

const { message: decrypted, corpId } = wecomCrypto.decrypt(encrypted);
console.log(`  Decrypted matches: ${decrypted === testMessage}`);
console.log(`  CorpID matches: ${corpId === CORP_ID}`);

if (decrypted !== testMessage) {
  console.log(`  FAIL: Expected "${testMessage.substring(0, 50)}..."`);
  console.log(`  Got: "${decrypted.substring(0, 50)}..."`);
} else {
  console.log("  PASS ✓");
}

// Test 2: Signature verification
console.log("\nTest 2: Signature computation and verification");
const timestamp = "1348831860";
const nonce = "123456789";

const signature = wecomCrypto.computeSignature(timestamp, nonce, encrypted);
console.log(`  Signature: ${signature}`);

const verified = wecomCrypto.verifySignature(signature, timestamp, nonce, encrypted);
console.log(`  Verification: ${verified}`);

const wrongSig = wecomCrypto.verifySignature("wrongsignature", timestamp, nonce, encrypted);
console.log(`  Wrong signature rejected: ${!wrongSig}`);

if (verified && !wrongSig) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

// Test 3: EchoStr decryption (URL verification flow)
console.log("\nTest 3: EchoStr decryption (URL verification)");
const echoStr = "test_echo_string_12345";
const encryptedEcho = wecomCrypto.encrypt(echoStr);
const echoSig = wecomCrypto.computeSignature(timestamp, nonce, encryptedEcho);

const decryptedEcho = wecomCrypto.decryptEchoStr(echoSig, timestamp, nonce, encryptedEcho);
console.log(`  Decrypted echo: "${decryptedEcho}"`);
console.log(`  Matches: ${decryptedEcho === echoStr}`);

if (decryptedEcho === echoStr) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

// Test 4: Full message decrypt flow
console.log("\nTest 4: Full message decrypt flow");
const msgEncrypted = wecomCrypto.encrypt(testMessage);
const msgSig = wecomCrypto.computeSignature(timestamp, nonce, msgEncrypted);

const decryptedMsg = wecomCrypto.decryptMessage(msgSig, timestamp, nonce, msgEncrypted);
console.log(`  Decrypted message matches: ${decryptedMsg === testMessage}`);

if (decryptedMsg === testMessage) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

// Test 5: XML parsing
console.log("\nTest 5: XML parsing");

const callbackXml = `<xml>
<ToUserName><![CDATA[wwc6c7670ccb97416b]]></ToUserName>
<AgentID><![CDATA[1000002]]></AgentID>
<Encrypt><![CDATA[${msgEncrypted}]]></Encrypt>
</xml>`;

const parsed = parseCallbackXml(callbackXml);
console.log(`  ToUserName: ${parsed.ToUserName}`);
console.log(`  AgentID: ${parsed.AgentID}`);
console.log(`  Encrypt length: ${parsed.Encrypt.length}`);
console.log(`  Encrypt matches: ${parsed.Encrypt === msgEncrypted}`);

const msgXml = parseMessageXml(testMessage);
console.log(`  FromUserName: ${msgXml.FromUserName}`);
console.log(`  MsgType: ${msgXml.MsgType}`);
console.log(`  Content: ${msgXml.Content}`);
console.log(`  MsgId: ${msgXml.MsgId}`);

if (
  parsed.ToUserName === "wwc6c7670ccb97416b" &&
  parsed.AgentID === "1000002" &&
  msgXml.FromUserName === "testuser" &&
  msgXml.Content === "Hello OpenClaw!"
) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

// Test 6: Reply XML generation
console.log("\nTest 6: Reply XML generation");
const replyXml = wecomCrypto.generateReplyXml(
  "<xml><Content>Reply from bot</Content></xml>",
  timestamp,
  nonce
);
console.log(`  Reply XML generated: ${replyXml.length > 0}`);
console.log(`  Contains Encrypt: ${replyXml.includes("<Encrypt>")}`);
console.log(`  Contains MsgSignature: ${replyXml.includes("<MsgSignature>")}`);

// Verify the reply can be decrypted
const replyParsed = parseCallbackXml(replyXml);
const replyDecrypted = wecomCrypto.decryptMessage(
  extractCData(replyXml, "MsgSignature"),
  timestamp,
  nonce,
  replyParsed.Encrypt
);
console.log(`  Reply decryptable: ${replyDecrypted !== null}`);

if (replyDecrypted && replyDecrypted.includes("Reply from bot")) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

// Test 7: Chinese characters
console.log("\nTest 7: Chinese character handling");
const chineseMsg = '<xml><Content><![CDATA[你好，OpenClaw！这是一条中文测试消息。]]></Content></xml>';
const chineseEncrypted = wecomCrypto.encrypt(chineseMsg);
const { message: chineseDecrypted } = wecomCrypto.decrypt(chineseEncrypted);
console.log(`  Chinese message roundtrip: ${chineseDecrypted === chineseMsg}`);

if (chineseDecrypted === chineseMsg) {
  console.log("  PASS ✓");
} else {
  console.log("  FAIL");
}

console.log("\n=== All Tests Complete ===");
