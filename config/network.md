# Network Topology

## Nodes

| Node | Role | WiFi / OOB | NIC1 (enp1s0f1np1) | NIC2 (enP2p1s0f1np1) |
|------|------|-----------|---------------------|----------------------|
| spark-2 | HEAD | 10.117.1.215 | 192.168.100.2 | 192.168.101.2 |
| spark-1 | WORKER | — | 192.168.100.1 | 192.168.101.1 |

## Physical Links

Both NICs on each node are connected via direct QSFP56 copper cable to the corresponding NIC on the peer node:

```
spark-2 enp1s0f1np1  (192.168.100.2)  <--200Gbps--> enp1s0f1np1  (192.168.100.1) spark-1
spark-2 enP2p1s0f1np1 (192.168.101.2) <--200Gbps--> enP2p1s0f1np1 (192.168.101.1) spark-1
```

Total raw bandwidth: 400 Gbps bidirectional.

## RoCE v2

Both NICs are configured with RoCE v2 ACTIVE.

- GID index: **3** (RoCEv2 — confirmed via `show_gids`)
- NCCL HCA string: `rocep1s0f1:1,roceP2p1s0f1:1`

NCCL env for dual-rail RoCE:
```
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1
```

## NFS

spark-2 exports `/home/sunil/models` to spark-1 over the 192.168.100.0/24 link:

```
# /etc/exports on spark-2
/home/sunil/models  192.168.100.1(ro,no_root_squash,async,no_subtree_check)
```

spark-1 mounts it at the same path:
```
# /etc/fstab on spark-1
192.168.100.2:/home/sunil/models  /home/sunil/models  nfs  ro,hard,intr,rsize=1048576,wsize=1048576  0 0
```

Both nodes see `/home/sunil/models/DeepSeek-V4-Flash-abliterated` at the same path, which is why the worker script uses the same volume mount path without modification.

## Gloo hostname binding

spark-2's `/etc/hosts` contains `127.0.1.1 dgx-spark-2`. Gloo (used for multi-node process group init in MP mode) resolves the hostname and binds to 127.0.1.1, making cross-node rendezvous impossible. Fix applied in `launch_head.sh`:

```
--add-host dgx-spark-2:10.117.1.215
```

This overrides the hosts entry inside the container so Gloo binds to the routable WiFi IP.
