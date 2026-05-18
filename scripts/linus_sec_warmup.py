#!/usr/bin/env python3
"""
linus-sec specific warmup. Emulates the exact request shape the harness sends:
- ~2.5k token system prompt (default.txt + env block)
- 47 tools (representative subset)
- streaming
- tool_choice=auto
- max_tokens varies from small to large

Compiles the kernel shapes the harness will actually hit.
"""
import sys, json, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
sys.stdout.reconfigure(line_buffering=True)

# Approximation of linus-sec default system prompt (~2500 tokens)
SYSTEM_PROMPT = """You are linus-sec, an interactive CLI tool for cyber security operations and software engineering tasks. Your job is to help users with security testing, vulnerability research, exploit development, and defensive operations.

IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

# System
- All text you output outside of tool use is displayed to the user. Output text to communicate with the user. You can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
- Tools are executed in a user-selected permission mode. When you attempt to call a tool that is not automatically allowed by the user's permission mode or permission settings, the user will be prompted so that they can approve or deny the execution.
- Tool results and user messages may include <system-reminder> or other tags. Tags contain information from the system. They bear no direct relation to the specific tool results or user messages in which they appear.

# Doing tasks
- The user will primarily request you to perform software engineering and security tasks. These may include solving bugs, adding new functionality, refactoring code, explaining code, conducting security assessments, building exploits, doing CTF challenges, and more.
- You are highly capable and often allow users to complete ambitious tasks that would otherwise be too complex or take too long.
- For exploratory questions ("what could we do about X?", "how should we approach this?", "what do you think?"), respond in 2-3 sentences with a recommendation and the main tradeoff.
- Prefer editing existing files to creating new ones.
- Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities unless explicitly authorized for security testing purposes.
- Default to writing no comments. Only add one when the WHY is non-obvious.

# Executing actions with care
Carefully consider the reversibility and blast radius of actions. Generally you can freely take local, reversible actions like editing files or running tests. But for actions that are hard to reverse, affect shared systems beyond your local environment, or could otherwise be risky or destructive, check with the user before proceeding.

# Tone and style
- Only use emojis if the user explicitly requests it.
- Your responses should be short and concise.
- When referencing specific functions or pieces of code include the pattern file_path:line_number to allow the user to easily navigate to the source code location.

# Using your tools
- Prefer dedicated tools over Bash when one fits.
- Use task management tools to plan and track work.
- You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel.

# Security operations specific
- Always verify the scope of authorization before conducting any security testing.
- Document findings with sufficient detail for remediation.
- Prefer non-destructive proof-of-concept over destructive exploitation.
- When working with sensitive data, sanitize logs and outputs.
- Respect rate limits and avoid causing denial of service unintentionally.

# Environment
You have been invoked in the following environment:
- Primary working directory: /Users/user/project
- Is a git repository: true
- Platform: darwin
- Shell: zsh
- OS Version: Darwin 25.4.0

When making function calls using tools that accept array or object parameters ensure those are structured using JSON. For example:
<example>
Tool call with structured parameters - parameters must be valid JSON objects matching the tool schema.
</example>

If you intend to call multiple tools and there are no dependencies between the calls, make all of the independent calls in the same response block, otherwise you MUST wait for previous calls to finish first to determine the dependent values."""

# 47-ish tools, representative of linus-sec registry
def make_tools():
    tools = []
    tool_defs = [
        ("bash", "Execute bash commands on the system", {"command": "string", "timeout": "integer"}),
        ("read", "Read file contents", {"path": "string", "offset": "integer", "limit": "integer"}),
        ("write", "Write content to a file", {"path": "string", "content": "string"}),
        ("edit", "Edit a file by replacing strings", {"path": "string", "old_string": "string", "new_string": "string", "replace_all": "boolean"}),
        ("glob", "Find files by pattern", {"pattern": "string", "path": "string"}),
        ("grep", "Search file contents", {"pattern": "string", "path": "string", "output_mode": "string"}),
        ("task", "Track or manage tasks", {"action": "string", "id": "string", "description": "string"}),
        ("fetch", "Fetch a URL", {"url": "string", "method": "string", "headers": "object"}),
        ("search", "Search for files or content", {"query": "string", "path": "string"}),
        ("code", "Execute code", {"language": "string", "source": "string"}),
        ("skill", "Invoke a skill", {"skill": "string", "args": "string"}),
        ("patch", "Apply a patch", {"patch": "string", "path": "string"}),
        ("question", "Ask the user a question", {"question": "string", "options": "array"}),
        ("httprequest", "Make an HTTP request", {"url": "string", "method": "string", "body": "string"}),
        ("browser", "Control a browser", {"action": "string", "url": "string", "selector": "string"}),
        ("interact", "Interactive shell", {"command": "string", "input": "string"}),
        ("workspace", "Workspace operations", {"action": "string", "name": "string"}),
        ("scope", "Define engagement scope", {"target": "string", "rules": "string"}),
        ("autonomy", "Set autonomy level", {"level": "string"}),
        ("evidence", "Record evidence", {"finding": "string", "data": "string"}),
        ("finding", "Document a finding", {"title": "string", "severity": "string", "description": "string"}),
        ("report", "Generate a report", {"format": "string", "output": "string"}),
        ("runbook", "Execute a runbook", {"name": "string", "args": "object"}),
        ("analyze", "Analyze data or code", {"target": "string", "type": "string"}),
        ("knowledge", "Query knowledge base", {"query": "string"}),
        ("scanner", "Run a scanner", {"tool": "string", "target": "string", "options": "array"}),
        ("appsec_probe", "Application security probe", {"url": "string", "checks": "array"}),
        ("opsec", "Operational security check", {"action": "string"}),
        ("remediate", "Apply remediation", {"finding_id": "string", "strategy": "string"}),
        ("cloud_posture", "Check cloud posture", {"provider": "string", "account": "string"}),
        ("container_surface", "Container attack surface", {"image": "string"}),
        ("iac_triage", "Infrastructure-as-code triage", {"path": "string"}),
        ("binary_triage", "Binary triage", {"path": "string", "arch": "string"}),
        ("crypto", "Cryptographic operations", {"operation": "string", "data": "string"}),
        ("net", "Network operations", {"command": "string", "target": "string"}),
        ("vault", "Vault operations", {"action": "string", "key": "string"}),
        ("doctor", "Self-diagnostic", {}),
        ("cve", "CVE lookup", {"id": "string"}),
        ("play", "Run a playbook", {"name": "string", "args": "object"}),
        ("pwn_bootstrap", "Bootstrap exploitation", {"target": "string"}),
        ("share", "Share an artifact", {"path": "string", "target": "string"}),
        ("methodology", "Methodology guidance", {"phase": "string"}),
        ("todo", "Manage todos", {"action": "string", "item": "string"}),
        ("metadata", "Get metadata", {"target": "string"}),
        ("compliance", "Compliance check", {"framework": "string", "target": "string"}),
        ("playbook", "Playbook orchestration", {"name": "string"}),
        ("notify", "Send a notification", {"channel": "string", "message": "string"}),
    ]
    for name, desc, params in tool_defs:
        props = {}
        required = []
        for pname, ptype in params.items():
            if ptype == "string": props[pname] = {"type": "string"}
            elif ptype == "integer": props[pname] = {"type": "integer"}
            elif ptype == "boolean": props[pname] = {"type": "boolean"}
            elif ptype == "array": props[pname] = {"type": "array", "items": {"type": "string"}}
            elif ptype == "object": props[pname] = {"type": "object"}
            required.append(pname)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required[:1] if required else []}
            }
        })
    return tools

TOOLS = make_tools()
print(f"Built {len(TOOLS)} tools, total size: {len(json.dumps(TOOLS))} bytes")

def wait_for_server(max_wait=600):
    print("Waiting for server...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"Ready after {time.time()-t0:.0f}s\n")
                    return True
        except Exception: pass
        time.sleep(10)
    return False

def call_streaming(user_msg, max_tokens=1000, label=""):
    """Match linus-sec's streaming request shape."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    payload = {
        "model": "deepseek-v4-flash",
        "messages": msgs,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    t0 = time.time()
    ttft = None
    total_chars = 0
    chunks = 0
    try:
        req = urllib.request.Request(f"{BASE}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]": break
                    try:
                        d = json.loads(chunk)
                        delta = d["choices"][0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if delta.get("tool_calls"): chunks += 1
                        if content and ttft is None:
                            ttft = time.time() - t0
                        if content: total_chars += len(content)
                        chunks += 1
                    except: pass
        elapsed = time.time() - t0
        print(f"  {label:30s}  ttft={ttft if ttft else 0:.2f}s  total={elapsed:6.1f}s  chunks={chunks}  chars={total_chars}", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {label:30s}  FAIL t={elapsed:.1f}s: {str(e)[:120]}", flush=True)
        return False

if not wait_for_server(): sys.exit(1)

print("="*70)
print("  linus-sec shape warmup - streaming + 47 tools + 2.5k system prompt")
print("="*70)

# Progressive warmup, increasing output sizes
tests = [
    ("simple question",         "What is 2+2?",                                            100),
    ("tool call short",         "Check the current directory contents.",                   500),
    ("tool call long",          "List files, then read the README.md.",                    1500),
    ("analysis short",          "Explain how TLS handshake works.",                        500),
    ("analysis medium",         "Compare BGP and OSPF routing protocols in detail.",       2000),
    ("code generation",         "Write a Python script to scan ports.",                    3000),
    ("long form essay",         "Write a detailed analysis of buffer overflow attacks.",   5000),
    ("very long output",        "Implement a full RESTful API in Python with auth.",       8000),
    ("max output test",         "Write a comprehensive book chapter about cryptography.", 16000),
]

ok = fail = 0
for label, msg, max_tok in tests:
    if call_streaming(msg, max_tokens=max_tok, label=label):
        ok += 1
    else:
        fail += 1
        print(f"  STOPPING after failure", flush=True)
        break

print()
print("="*70)
print(f"  linus-sec warmup: {ok}/{len(tests)} ok, {fail} failed")
print("="*70)
