/**
 * Simple XML parser for WeCom callback messages
 * Avoids heavy xml2js dependency by using regex-based parsing
 * for the well-defined WeCom XML message format
 */

/**
 * Parse WeCom callback XML body
 * Format:
 * <xml>
 *   <ToUserName><![CDATA[toUser]]></ToUserName>
 *   <AgentID><![CDATA[toAgentID]]></AgentID>
 *   <Encrypt><![CDATA[msg_encrypt]]></Encrypt>
 * </xml>
 */
export function parseCallbackXml(xml: string): {
  ToUserName: string;
  AgentID: string;
  Encrypt: string;
} {
  return {
    ToUserName: extractCData(xml, "ToUserName"),
    AgentID: extractCData(xml, "AgentID"),
    Encrypt: extractCData(xml, "Encrypt"),
  };
}

/**
 * Parse decrypted message XML
 * Format varies by MsgType, common fields:
 * <xml>
 *   <ToUserName><![CDATA[corpid]]></ToUserName>
 *   <FromUserName><![CDATA[userid]]></FromUserName>
 *   <CreateTime>1348831860</CreateTime>
 *   <MsgType><![CDATA[text]]></MsgType>
 *   <Content><![CDATA[hello]]></Content>
 *   <MsgId>1234567890123456</MsgId>
 *   <AgentID>1</AgentID>
 * </xml>
 */
export function parseMessageXml(xml: string): Record<string, string> {
  const result: Record<string, string> = {};

  // Match both CDATA and plain text values
  const cdataRegex = /<(\w+)><!\[CDATA\[(.*?)\]\]><\/\1>/gs;
  const plainRegex = /<(\w+)>([^<]+)<\/\1>/g;

  let match;

  // Extract CDATA values first
  while ((match = cdataRegex.exec(xml)) !== null) {
    result[match[1]] = match[2];
  }

  // Extract plain text values (numbers, etc.)
  while ((match = plainRegex.exec(xml)) !== null) {
    if (!(match[1] in result)) {
      result[match[1]] = match[2];
    }
  }

  return result;
}

/**
 * Extract CDATA content from an XML tag
 */
function extractCData(xml: string, tagName: string): string {
  // Try CDATA format first
  const cdataRegex = new RegExp(
    `<${tagName}><!\\[CDATA\\[([\\s\\S]*?)\\]\\]></${tagName}>`,
    "i"
  );
  const cdataMatch = xml.match(cdataRegex);
  if (cdataMatch) return cdataMatch[1];

  // Fall back to plain text
  const plainRegex = new RegExp(`<${tagName}>([^<]*)</${tagName}>`, "i");
  const plainMatch = xml.match(plainRegex);
  if (plainMatch) return plainMatch[1];

  return "";
}
