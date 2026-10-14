#!/usr/bin/env python3
"""Secret / PII scan over what would be published (git ls-files).

Usage:  python3 tools/scan-publish.py [--json OUT] [--all-files]

Generic rules (keys, tokens, private keys, Bearer values, e-mail addresses, routable IPv4,
absolute user paths) are defined here. Site-specific literals (domain, user names, e-mail,
public IPs) are NOT stored in this tracked file: they are read at run time from the
gitignored `config/local.env` (values of QWEN36_PUBLIC_HOST, QWEN36_FORGE_HOST,
QWEN36_WSL_USER, QWEN36_ROOT, QWEN36_WIN_ROOT, QWEN36_ARCHIVE, QWEN36_WORKSPACE*) and
`local/scan-literals.txt` (one literal per line). Allowed findings are listed with a
justification in `tools/scan-allowlist.txt` as `<path-glob> <rule> <justification>`.
Exit status 1 when any finding is not allowlisted.
"""
import argparse, fnmatch, ipaddress, json, pathlib, re, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parents[1]

GENERIC = {
    'private-key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'aws-key': r'\bAKIA[0-9A-Z]{16}\b',
    'github-token': r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'openai-style-key': r'\bsk-[A-Za-z0-9_-]{20,}\b',
    'hf-token': r'\bhf_[A-Za-z0-9]{30,}\b',
    'slack-token': r'\bxox[abprs]-[A-Za-z0-9-]{10,}\b',
    'jwt': r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.',
    'bearer-value': r'Bearer\s+[A-Za-z0-9._~+/=-]{24,}',
    'secret-assignment': r'(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*["\'][^"\'\s]{16,}["\']',
    'cloudflare-token-name': r'(?i)\bCF_API_TOKEN\s*=\s*\S+',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'home-path': r'/home/(?!USER\b|user\b|runner\b|\$|<)[A-Za-z0-9_.-]+',
    'windows-user-path': r'(?i)\b[A-Z]:\\{1,2}Users\\{1,2}(?!USER\b|Public\b|<)[A-Za-z0-9_.~-]+',
    'wsl-drive-path': r'/mnt/[a-z]/(?!Windows/)',
    'windows-site-root': r'(?i)\b[D-Z]:\\{1,2}(LLM|Codex|WSL|Projects)\b',
}
EMAIL_OK = re.compile(r'(?i)@(example\.(com|org|net)|[a-z0-9.-]+\.invalid|anthropic\.com|users\.noreply\.github\.com)$'
                      r'|^noreply@|^git@github\.com$|^user@[0-9]+\.service$')
IPV4 = re.compile(r'(?<![\w.-])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\w.-])')
TEXT_SUFFIX_SKIP = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.woff', '.woff2', '.so', '.whl'}


def literals():
    out = set()
    env = REPO / 'config/local.env'
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*(QWEN36_[A-Z_]+)\s*=\s*'?([^'#]*?)'?\s*(#.*)?$", line)
            if not m or not m.group(2):
                continue
            key, val = m.group(1), m.group(2).strip()
            if key in ('QWEN36_PUBLIC_HOST', 'QWEN36_FORGE_HOST', 'QWEN36_WSL_USER', 'QWEN36_ROOT',
                       'QWEN36_WIN_ROOT', 'QWEN36_ARCHIVE', 'QWEN36_WORKSPACE', 'QWEN36_WORKSPACE_WIN'):
                out.add(val)
                if key in ('QWEN36_PUBLIC_HOST', 'QWEN36_FORGE_HOST') and val.count('.') >= 2:
                    out.add(val.split('.', 1)[1])          # registrable domain
    extra = REPO / 'local/scan-literals.txt'
    if extra.exists():
        out |= {l.strip() for l in extra.read_text().splitlines() if l.strip() and not l.startswith('#')}
    return sorted((v for v in out if len(v) >= 4), key=len, reverse=True)


def routable(ip):
    try:
        a = ipaddress.IPv4Address(ip)
    except ValueError:
        return False
    if a.is_loopback or a.is_unspecified or a.is_multicast:
        return False
    for net in ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24', '1.1.1.1/32', '8.8.8.8/32', '255.255.255.255/32'):
        if a in ipaddress.ip_network(net):
            return False
    return True


def load_allow():
    rows = []
    p = REPO / 'tools/scan-allowlist.txt'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                glob, rule, *why = line.split(None, 2)
                rows.append((glob, rule, why[0] if why else ''))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    ap.add_argument('--all-files', action='store_true', help='scan the working tree incl. untracked (not ignored) files')
    args = ap.parse_args()
    cmd = ['git', '-C', str(REPO), 'ls-files', '-z'] + (['--others', '--exclude-standard', '--cached'] if args.all_files else [])
    files = [f for f in subprocess.run(cmd, capture_output=True, check=True).stdout.decode().split('\0') if f]
    lits = literals()
    rules = {k: re.compile(v) for k, v in GENERIC.items()}
    allow = load_allow()
    findings, scanned = [], 0
    for rel in files:
        p = REPO / rel
        if not p.is_file() or p.suffix.lower() in TEXT_SUFFIX_SKIP:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append({'file': rel, 'line': 0, 'rule': 'binary-file', 'excerpt': ''})
            continue
        scanned += 1
        for no, line in enumerate(text.splitlines(), 1):
            hits = []
            for name, rx in rules.items():
                for m in rx.finditer(line):
                    if name == 'email' and EMAIL_OK.search(m.group(0)):
                        continue
                    hits.append((name, m.group(0)))
            for m in IPV4.finditer(line):
                if routable(m.group(1)):
                    hits.append(('routable-ipv4', m.group(1)))
            low = line.lower()
            for v in lits:
                if v.lower() in low:
                    hits.append(('site-literal', v))
            for name, val in hits:
                if any(fnmatch.fnmatch(rel, g) and (r == name or r == '*') for g, r, _ in allow):
                    continue
                masked = val if name in ('routable-ipv4', 'home-path', 'wsl-drive-path', 'windows-site-root') else val[:3] + '***'
                findings.append({'file': rel, 'line': no, 'rule': name, 'excerpt': masked})
    report = {'files_listed': len(files), 'text_files_scanned': scanned, 'site_literals_checked': len(lits),
              'generic_rules': sorted(GENERIC) + ['routable-ipv4'], 'findings': findings,
              'allowlist': [{'glob': g, 'rule': r, 'why': w} for g, r, w in allow]}
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1))
    for f in findings[:200]:
        print(f"{f['file']}:{f['line']}: {f['rule']}: {f['excerpt']}")
    print(f"scan: {len(files)} files listed, {scanned} text files, {len(lits)} site literals, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
