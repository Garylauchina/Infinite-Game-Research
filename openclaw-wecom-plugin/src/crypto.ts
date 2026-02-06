/**
 * WeCom (企业微信) Message Encryption & Decryption
 *
 * Implements the official WeCom callback message encryption protocol:
 * - SHA1 signature verification
 * - AES-256-CBC encryption/decryption with PKCS#7 padding
 * - XML message parsing
 */

import crypto from "node:crypto";

/**
 * WeCom message crypto helper
 */
export class WeComCrypto {
  private token: string;
  private encodingAESKey: string;
  private aesKey: Buffer;
  private iv: Buffer;
  private corpId: string;

  constructor(token: string, encodingAESKey: string, corpId: string) {
    this.token = token;
    this.encodingAESKey = encodingAESKey;
    this.corpId = corpId;

    // EncodingAESKey is Base64 encoded, decode to get the actual AES key (32 bytes)
    this.aesKey = Buffer.from(encodingAESKey + "=", "base64");
    // IV is the first 16 bytes of the AES key
    this.iv = this.aesKey.subarray(0, 16);
  }

  /**
   * Verify the message signature from WeCom
   * signature = SHA1(sort(token, timestamp, nonce, encrypt))
   */
  verifySignature(
    msgSignature: string,
    timestamp: string,
    nonce: string,
    encrypt: string
  ): boolean {
    const computed = this.computeSignature(timestamp, nonce, encrypt);
    return computed === msgSignature;
  }

  /**
   * Compute SHA1 signature
   * Sort token, timestamp, nonce, encrypt alphabetically, then SHA1
   */
  computeSignature(
    timestamp: string,
    nonce: string,
    encrypt: string
  ): string {
    const parts = [this.token, timestamp, nonce, encrypt].sort();
    const str = parts.join("");
    return crypto.createHash("sha1").update(str).digest("hex");
  }

  /**
   * Decrypt an encrypted message from WeCom
   * Returns the decrypted XML message content
   */
  decrypt(encrypted: string): { message: string; corpId: string } {
    // Base64 decode the encrypted string
    const encryptedBuffer = Buffer.from(encrypted, "base64");

    // AES-256-CBC decrypt
    const decipher = crypto.createDecipheriv(
      "aes-256-cbc",
      this.aesKey,
      this.iv
    );
    decipher.setAutoPadding(false);

    let decrypted = Buffer.concat([
      decipher.update(encryptedBuffer),
      decipher.final(),
    ]);

    // Remove PKCS#7 padding
    const padLen = decrypted[decrypted.length - 1];
    decrypted = decrypted.subarray(0, decrypted.length - padLen);

    // Parse the decrypted content:
    // random(16 bytes) + msg_len(4 bytes, network byte order) + msg + receiveid
    const random = decrypted.subarray(0, 16);
    const msgLen = decrypted.readUInt32BE(16);
    const message = decrypted.subarray(20, 20 + msgLen).toString("utf-8");
    const corpId = decrypted.subarray(20 + msgLen).toString("utf-8");

    return { message, corpId };
  }

  /**
   * Encrypt a message for WeCom (used for passive reply)
   */
  encrypt(message: string): string {
    // random(16 bytes) + msg_len(4 bytes) + msg + corpId
    const random = crypto.randomBytes(16);
    const msgBuffer = Buffer.from(message, "utf-8");
    const msgLen = Buffer.alloc(4);
    msgLen.writeUInt32BE(msgBuffer.length, 0);
    const corpIdBuffer = Buffer.from(this.corpId, "utf-8");

    let plaintext = Buffer.concat([random, msgLen, msgBuffer, corpIdBuffer]);

    // PKCS#7 padding
    const blockSize = 32;
    const padLen = blockSize - (plaintext.length % blockSize);
    const padding = Buffer.alloc(padLen, padLen);
    plaintext = Buffer.concat([plaintext, padding]);

    // AES-256-CBC encrypt
    const cipher = crypto.createCipheriv(
      "aes-256-cbc",
      this.aesKey,
      this.iv
    );
    cipher.setAutoPadding(false);

    const encrypted = Buffer.concat([
      cipher.update(plaintext),
      cipher.final(),
    ]);

    return encrypted.toString("base64");
  }

  /**
   * Generate encrypted reply XML
   */
  generateReplyXml(
    replyMsg: string,
    timestamp: string,
    nonce: string
  ): string {
    const encrypt = this.encrypt(replyMsg);
    const signature = this.computeSignature(timestamp, nonce, encrypt);

    return `<xml>
<Encrypt><![CDATA[${encrypt}]]></Encrypt>
<MsgSignature><![CDATA[${signature}]]></MsgSignature>
<TimeStamp>${timestamp}</TimeStamp>
<Nonce><![CDATA[${nonce}]]></Nonce>
</xml>`;
  }

  /**
   * Decrypt and verify an echostr (for URL verification)
   */
  decryptEchoStr(
    msgSignature: string,
    timestamp: string,
    nonce: string,
    echostr: string
  ): string | null {
    // Verify signature
    if (!this.verifySignature(msgSignature, timestamp, nonce, echostr)) {
      return null;
    }

    // Decrypt echostr
    const { message } = this.decrypt(echostr);
    return message;
  }

  /**
   * Decrypt and verify a callback message
   */
  decryptMessage(
    msgSignature: string,
    timestamp: string,
    nonce: string,
    encrypt: string
  ): string | null {
    // Verify signature
    if (!this.verifySignature(msgSignature, timestamp, nonce, encrypt)) {
      return null;
    }

    // Decrypt message
    const { message, corpId } = this.decrypt(encrypt);

    // Verify corpId
    if (corpId !== this.corpId) {
      return null;
    }

    return message;
  }
}
