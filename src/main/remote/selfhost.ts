/**
 * The bring-your-own-domain path: FreeDNS dynamic DNS + in-process Let's Encrypt.
 *
 * WACS (win-acme) was investigated and rejected: it needs administrator rights, registers a
 * SYSTEM scheduled task, and that task invokes wacs.exe by absolute path — so it breaks every
 * time this app is moved, which is a thing this app is explicitly designed to do.
 *
 * `acme-client` runs in-process instead: no admin, no external binary, no scheduled task, and
 * renewal happens on a timer while the app is running. Same Let's Encrypt certificates.
 *
 * HTTP-01 needs port 80 reachable from the internet, which means the user must forward it.
 * The guidance for that is in the UI; the mechanics are here.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import http from 'node:http'
import { EventEmitter } from 'node:events'
import acme from 'acme-client'
import { APPDATA_DIR } from '../storage/paths'
import { getFreednsToken } from './auth'

const CERT_DIR = path.join(APPDATA_DIR, 'certs')
const ACCOUNT_KEY_FILE = path.join(CERT_DIR, 'account.key')

export interface CertificateInfo {
  domain: string
  certPath: string
  keyPath: string
  expiresAt: number
  issuedAt: number
}

export interface DdnsResult {
  ok: boolean
  ip?: string
  message: string
}

// ---------------------------------------------------------------- dynamic DNS

/**
 * FreeDNS (freedns.afraid.org) update.
 *
 * Their scheme is a per-record update URL containing an opaque token; hitting it points the
 * record at the requesting IP. We resolve our own public address first so the result can be
 * reported honestly rather than just echoing "updated".
 */
export async function publicIp(): Promise<string | null> {
  const sources = ['https://api.ipify.org', 'https://ifconfig.me/ip', 'https://icanhazip.com']
  for (const url of sources) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(6000) })
      if (!res.ok) continue
      const ip = (await res.text()).trim()
      if (/^\d{1,3}(\.\d{1,3}){3}$/.test(ip)) return ip
    } catch {
      /* try the next source */
    }
  }
  return null
}

export async function updateFreedns(token?: string | null): Promise<DdnsResult> {
  const key = token ?? getFreednsToken()
  if (!key) {
    return { ok: false, message: 'No FreeDNS update token set. Get one from freedns.afraid.org (Dynamic DNS -> Direct URL).' }
  }

  const url = key.startsWith('http') ? key : `https://freedns.afraid.org/dynamic/update.php?${key}`
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) })
    const text = (await res.text()).trim()
    const ip = await publicIp()

    // FreeDNS answers in prose; "Updated" and "no IP change" are both success.
    const ok = res.ok && /updated|no ip change|has not changed/i.test(text)
    return {
      ok,
      ip: ip ?? undefined,
      message: ok ? `${text}${ip ? ` (public IP ${ip})` : ''}` : `FreeDNS said: ${text}`
    }
  } catch (err) {
    return { ok: false, message: err instanceof Error ? err.message : String(err) }
  }
}

/** Check whether the domain actually resolves to this machine before attempting ACME. */
export async function verifyDomainPointsHere(domain: string): Promise<{ ok: boolean; message: string }> {
  const [ours, theirs] = await Promise.all([publicIp(), resolveA(domain)])
  if (!ours) return { ok: false, message: 'Could not determine this machine\'s public IP.' }
  if (!theirs.length) return { ok: false, message: `${domain} does not resolve to any A record yet. DNS may still be propagating.` }
  if (!theirs.includes(ours)) {
    return {
      ok: false,
      message: `${domain} points at ${theirs.join(', ')} but this machine is ${ours}. Update the DNS record (or run a DDNS update) and wait for propagation.`
    }
  }
  return { ok: true, message: `${domain} correctly resolves to ${ours}.` }
}

async function resolveA(domain: string): Promise<string[]> {
  try {
    // DNS-over-HTTPS avoids depending on the local resolver's cache.
    const res = await fetch(`https://cloudflare-dns.com/dns-query?name=${encodeURIComponent(domain)}&type=A`, {
      headers: { Accept: 'application/dns-json' },
      signal: AbortSignal.timeout(8000)
    })
    if (!res.ok) return []
    const json = (await res.json()) as { Answer?: { type: number; data: string }[] }
    return (json.Answer ?? []).filter((a) => a.type === 1).map((a) => a.data)
  } catch {
    return []
  }
}

// ---------------------------------------------------------------- ACME

class CertificateManager extends EventEmitter {
  private challengeServer: http.Server | null = null
  private challenges = new Map<string, string>()
  private renewTimer: NodeJS.Timeout | null = null

  /** Serve /.well-known/acme-challenge/* on port 80 for the duration of the order. */
  private async startChallengeServer(): Promise<void> {
    if (this.challengeServer) return
    const server = http.createServer((req, res) => {
      const match = req.url?.match(/^\/\.well-known\/acme-challenge\/(.+)$/)
      if (match) {
        const value = this.challenges.get(match[1])
        if (value) {
          res.writeHead(200, { 'Content-Type': 'text/plain' })
          res.end(value)
          return
        }
      }
      res.writeHead(404)
      res.end()
    })

    await new Promise<void>((resolve, reject) => {
      server.once('error', (err: NodeJS.ErrnoException) => {
        reject(
          new Error(
            err.code === 'EADDRINUSE'
              ? 'Port 80 is already in use. Stop the other web server, or use the tunnel instead.'
              : `Could not bind port 80: ${err.message}`
          )
        )
      })
      server.listen(80, '0.0.0.0', () => resolve())
    })
    this.challengeServer = server
  }

  private async stopChallengeServer(): Promise<void> {
    const server = this.challengeServer
    this.challengeServer = null
    if (server) await new Promise<void>((resolve) => server.close(() => resolve()))
  }

  private async accountKey(): Promise<Buffer> {
    await fsp.mkdir(CERT_DIR, { recursive: true })
    try {
      return await fsp.readFile(ACCOUNT_KEY_FILE)
    } catch {
      const key = await acme.crypto.createPrivateKey()
      await fsp.writeFile(ACCOUNT_KEY_FILE, key)
      return key
    }
  }

  paths(domain: string): { certPath: string; keyPath: string } {
    const safe = domain.replace(/[^a-z0-9.-]/gi, '_')
    return {
      certPath: path.join(CERT_DIR, `${safe}.crt`),
      keyPath: path.join(CERT_DIR, `${safe}.key`)
    }
  }

  existing(domain: string): CertificateInfo | null {
    const { certPath, keyPath } = this.paths(domain)
    if (!fs.existsSync(certPath) || !fs.existsSync(keyPath)) return null
    try {
      const cert = fs.readFileSync(certPath, 'utf8')
      const info = acme.crypto.readCertificateInfo(cert)
      return {
        domain,
        certPath,
        keyPath,
        issuedAt: info.notBefore.getTime(),
        expiresAt: info.notAfter.getTime()
      }
    } catch {
      return null
    }
  }

  /**
   * Issue or renew a certificate. Reuses a valid one unless `force`, so calling this on every
   * launch is cheap and safe.
   */
  async obtain(domain: string, email: string, opts: { force?: boolean; staging?: boolean } = {}): Promise<CertificateInfo> {
    const existing = this.existing(domain)
    const THIRTY_DAYS = 30 * 24 * 60 * 60 * 1000
    if (existing && !opts.force && existing.expiresAt - Date.now() > THIRTY_DAYS) {
      return existing
    }

    const check = await verifyDomainPointsHere(domain)
    if (!check.ok) throw new Error(check.message)

    await this.startChallengeServer()
    this.emit('progress', 'Requesting certificate from Let\'s Encrypt…')

    try {
      const client = new acme.Client({
        directoryUrl: opts.staging ? acme.directory.letsencrypt.staging : acme.directory.letsencrypt.production,
        accountKey: await this.accountKey()
      })

      const [key, csr] = await acme.crypto.createCsr({ commonName: domain })

      const cert = await client.auto({
        csr,
        email,
        termsOfServiceAgreed: true,
        challengePriority: ['http-01'],
        challengeCreateFn: async (_authz, challenge, keyAuthorization) => {
          this.challenges.set(challenge.token, keyAuthorization)
        },
        challengeRemoveFn: async (_authz, challenge) => {
          this.challenges.delete(challenge.token)
        }
      })

      const { certPath, keyPath } = this.paths(domain)
      await fsp.mkdir(CERT_DIR, { recursive: true })
      await fsp.writeFile(certPath, cert)
      await fsp.writeFile(keyPath, key)

      const info = acme.crypto.readCertificateInfo(cert)
      const result: CertificateInfo = {
        domain,
        certPath,
        keyPath,
        issuedAt: info.notBefore.getTime(),
        expiresAt: info.notAfter.getTime()
      }
      this.emit('issued', result)
      return result
    } finally {
      await this.stopChallengeServer()
    }
  }

  /**
   * Renewal timer. Runs in-process, which is the whole point: no scheduled task to go stale
   * when the exe moves.
   */
  scheduleRenewal(domain: string, email: string): void {
    this.clearRenewal()
    const check = async (): Promise<void> => {
      try {
        const existing = this.existing(domain)
        const THIRTY_DAYS = 30 * 24 * 60 * 60 * 1000
        if (!existing || existing.expiresAt - Date.now() < THIRTY_DAYS) {
          this.emit('progress', 'Renewing certificate…')
          await this.obtain(domain, email, { force: true })
        }
      } catch (err) {
        this.emit('error', err instanceof Error ? err.message : String(err))
      }
    }
    // Twice daily is what every ACME client does; it is cheap and tolerates downtime.
    this.renewTimer = setInterval(() => void check(), 12 * 60 * 60 * 1000)
    void check()
  }

  clearRenewal(): void {
    if (this.renewTimer) clearInterval(this.renewTimer)
    this.renewTimer = null
  }
}

export const certificates = new CertificateManager()

/** Keep the DDNS record fresh while the app runs. */
let ddnsTimer: NodeJS.Timeout | null = null

export function startDdnsUpdates(intervalMinutes = 15): void {
  stopDdnsUpdates()
  ddnsTimer = setInterval(() => void updateFreedns(), intervalMinutes * 60 * 1000)
  void updateFreedns()
}

export function stopDdnsUpdates(): void {
  if (ddnsTimer) clearInterval(ddnsTimer)
  ddnsTimer = null
}
