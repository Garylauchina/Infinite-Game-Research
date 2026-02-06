/**
 * WeCom Plugin Runtime State
 *
 * Stores references to the OpenClaw runtime and API clients
 * that are initialized during plugin registration.
 */

import type { WeComApiClient } from "./api.js";
import type { WeComCrypto } from "./crypto.js";

let _runtime: any = null;
let _logger: any = console;
let _apiClients: Map<string, WeComApiClient> = new Map();
let _cryptoInstances: Map<string, WeComCrypto> = new Map();

export function setRuntime(runtime: any) {
  _runtime = runtime;
}

export function getRuntime(): any {
  return _runtime;
}

export function setLogger(logger: any) {
  _logger = logger;
}

export function getLogger(): any {
  return _logger;
}

export function setApiClient(accountId: string, client: WeComApiClient) {
  _apiClients.set(accountId, client);
}

export function getApiClient(accountId: string): WeComApiClient | undefined {
  return _apiClients.get(accountId);
}

export function setCryptoInstance(accountId: string, instance: WeComCrypto) {
  _cryptoInstances.set(accountId, instance);
}

export function getCryptoInstance(accountId: string): WeComCrypto | undefined {
  return _cryptoInstances.get(accountId);
}

export function getAllAccountIds(): string[] {
  return Array.from(_apiClients.keys());
}
