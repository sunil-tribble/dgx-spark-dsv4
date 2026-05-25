# Dual-instance MiniMax-M2.7 setup (production)

Following the v3 bench finding that `--reasoning off` is decisive for everything
*except* math (where reasoning-on caught arithmetic errors), production runs
two instances across the dual-Spark cluster.

## Topology

| Endpoint | Spark | Port | `--reasoning` | ctx | Use case |
|---|---|---:|---|---:|---|
| `minimax-m2.7` | spark-2 (10.117.1.215) | 8096 | **off** + budget 0 | 131072 | Default for all agentic workloads |
| `minimax-m2.7-math` | spark-1 (10.117.1.24) | 8097 | **auto** + budget 4096 | 32768 | Math, proofs, multi-step derivations |

Both load the same `UD-IQ4_XS` GGUF (108 GB weights). Each Spark has its
own copy of the model — no cross-node sharing (NFS would be too slow for
mmap'd inference weights).

## Why dual-Spark not dual-instance-on-one-Spark

Tried first: two llama-server processes on spark-2 sharing weights via mmap
(dropped `--mlock`). Hit **CUDA OOM** on the second process — DGX Spark's
unified memory means CPU RAM = GPU VRAM, and when instance 1 holds 115 GB,
instance 2 can't get a CUDA context. Putting the math instance on spark-1
costs nothing because spark-1 was idle.

## SystemD units

Files in this directory:
- `llama-minimax27-default.service` — spark-2:8096
- `llama-minimax27-math.service` — spark-1:8097
- `launch_8096_noreason.sh` — spark-2 launcher
- `launch_8097_math.sh` — spark-1 launcher

Installed at `/etc/systemd/system/` on each respective node, enabled at boot,
restart-on-failure.

## Hermes / sparklinus integration

`/home/sunil/.hermes/config.yaml` on sunil-inference:
```yaml
providers:
  minimax-spark:
    base_url: http://10.117.1.215:8096/v1
  minimax-spark-math:
    base_url: http://10.117.1.24:8097/v1
```

Model selection at the gateway: explicit per-tool (e.g., a `solve_math`
tool routes to `minimax-spark-math`, default chat routes to
`minimax-spark`). The agent picks the model name; the gateway picks the
provider URL from the model→provider mapping in config.

## Smoke-tested behavior (2026-05-25)

| | 8096 (default) | 8097 (math) |
|---|---|---|
| `17*23?` → content | `391` | `391` |
| `17*23?` → reasoning_content | `""` (none) | `224 chars` (thought briefly) |
| Decode speed | 26.6 t/s | 22.5 t/s |
| Memory (host) | 115 GB | 108 GB |

## When to use which

| Workload | Endpoint |
|---|---|
| Tool use / agent chat / planning / code / synthesis / long-context recall / general Q&A | `minimax-m2.7` |
| Multi-step proofs, induction arguments, optimization, group theory, real analysis | `minimax-m2.7-math` |
| Quick factual recall | `minimax-m2.7` (faster) |
| Anything where wrong-but-confident is worse than no-answer | `minimax-m2.7-math` (will use reasoning to self-check) |

## Recovery

Both services have `Restart=always`. If a Spark reboots:
```bash
ssh sunil@10.117.1.215 'sudo systemctl status llama-minimax27-default'
ssh sunil@10.117.1.24  'sudo systemctl status llama-minimax27-math'
```

If sparklinus is wired to a stale endpoint after a service shift, restart hermes-gateway:
```bash
ssh sunil@10.117.1.229 'systemctl --user restart hermes-gateway'
```
