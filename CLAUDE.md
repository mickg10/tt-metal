# GLM-4.7 Full (355B) on Blackhole Galaxy — tt-metal workspace

## ACTIVE MODEL: GLM-4.7 Full (355B, 160 experts)

This workspace runs **TWO models** on BH Galaxy but the **current focus is 355B Full**:

| Model | Code | Env File | Container Project | Status |
|-------|------|----------|-------------------|--------|
| **GLM-4.7 Full (355B)** | `models/demos/glm4_moe/tt/` | `.env.glm47_reap_blackhole` | `reap-bh` | **ACTIVE** |
| GLM-4.7-Flash (47B) | `models/demos/glm4_moe_lite/tt/` | `.env.glm47_blackhole` | `bh1XX` | Done (10.1 tok/s) |

**DO NOT work on Flash unless explicitly told to.**

## CRITICAL: WE OWN TT-METAL — FIX KERNEL BUGS OURSELVES

**NEVER say "file a bug with TT engineering" or "wait for upstream fix."**
**We have full tt-metal source. When a kernel is broken, WE fix it:**
1. Write a C++ gtest that reproduces the failure
2. Root-cause the bug in C++/LLK source (use Codex+Gemini researchers)
3. Implement the fix
4. Verify with tests
5. Commit test + fix together

## CRITICAL: ALWAYS SOURCE BUILD — NEVER USE PREBUILT TTNN

**Every `.env` file MUST have `SKIP_TT_METAL_BUILD=0`.** The prebuilt `_ttnn.so` in the
Docker base image is from a DIFFERENT tt-metal commit than the bind-mounted source. Using
prebuilt causes silent bugs (broken SDPA decode, garbled output, missing Python bindings).

**Lesson learned (2026-03-17):** ~10 hours lost debugging garbled GLM-4.7 output on BH Galaxy.
Root cause was `SKIP_TT_METAL_BUILD=1` using a stale prebuilt binary. Source build fixed it.

**NEVER run `docker system prune -af`** — it destroys the Docker image, build volumes, and
venv cache, requiring hours of rebuild. Use `docker rm -f $(docker ps -aq)` for cleanup.

## CRITICAL: Remote Machine Rules

This workspace is the LOCAL source of truth. The Blackhole Galaxy machine
(`ssh -p 55212 mick@38.97.6.6`) is a SLAVE that receives code via rsync.

### FORBIDDEN on the remote BH machine:
- `git clone`, `git checkout`, `git pull`, `git fetch`, `git rebase`, `git commit`
- `git switch`, `git branch`, `git merge`, `git stash`, `git reset`
- ANY git command whatsoever — the remote has NO .git directory
- **`sed`, `echo >`, `cat >`, `Edit`, `Write` or ANY file modification** — NO editing code on remote!
- **ALL code changes MUST be made locally and rsynced** — the remote is a read-only slave
- Exception: env files (`.env.*`) and docker-compose files may be edited on remote for runtime config

### REQUIRED workflow (master → slave):
1. Edit files LOCALLY in this workspace (`ws/glm47_flash_blackhole_galaxy/tt-metal/`)
2. Rsync to remote:
   ```bash
   rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' \
     --exclude='build_Release' --exclude='python_env' --exclude='generated' \
     /home/ttuser/src_docker/ws/glm47_flash_blackhole_galaxy/tt-metal/ \
     mick@38.97.6.6:/home/mick/ws/glm47_flash_blackhole_galaxy/tt-metal/ \
     -e "ssh -p 55212"
   ```
3. Restart container on remote to pick up changes

### CRITICAL: Researchers/Agents MUST check REMOTE state
- **NEVER trust local files for runtime config.** The env file on the remote machine
  may differ from the local copy (implementers modify env directly on remote via sed).
- **ALWAYS SSH to remote to read the actual env file and container config:**
  ```bash
  ssh -p 55212 mick@38.97.6.6 "cat /home/mick/ws/glm47_flash_blackhole_galaxy/docker_tt/dev/.env.glm47_reap_blackhole"
  ssh -p 55212 mick@38.97.6.6 "docker inspect --format='{{.Config.Env}}' <container-name>"
  ssh -p 55212 mick@38.97.6.6 "docker logs --tail 20 <container-name>"
  ```
- **Check remote for runtime state:** container logs, loaded layer count, actual
  env vars inside container, device status, docker ps output.
- Reading only local files leads to WRONG conclusions (e.g., seeing NUM_LAYERS=1
  locally when remote has NUM_LAYERS=92).

### Container commands (355B — run ON REMOTE via SSH):
```bash
ssh -p 55212 mick@38.97.6.6
cd /home/mick/ws/glm47_flash_blackhole_galaxy/docker_tt
docker compose -p reap-bh --env-file dev/.env.glm47_reap_blackhole \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  -f dev/docker-compose.blackhole.yml \
  up -d vllm-tt
```

### Benchmark commands (run ON REMOTE via SSH):
```bash
# Inside the container or via docker exec:
python tests/bench_decode.py --url http://localhost:8088 --gen-tokens 500 \
  --only-batch 1 --skip-combined --prefill-contexts 0
```

## Key Model Code (355B Full = glm4_moe)
- `models/demos/glm4_moe/tt/moe_tt.py` — MoE routing, sparse matmul config
- `models/demos/glm4_moe/tt/decoder_layer_tt.py` — Forward pass
- `models/demos/glm4_moe/tt/layer_weights.py` — Weight loading
- `models/demos/glm4_moe/tt/attention_tt.py` — Attention (GQA, 96 heads)
- `models/demos/glm4_moe/tt/model_tt.py` — Decode loop, trace capture
- `models/demos/glm4_moe/tt/generator_vllm.py` — vLLM interface

## 355B Architecture
- 92 layers, 160 routed experts (top-8), 1 shared expert
- hidden=5120, 96 Q heads, 8 KV heads, head_dim=128
- FP8 weights (362 GB), ~14.75 GB/device
- TP=4, DP=8, EP=32, mesh (8,4)

## Blackhole Hardware
- 13×10 = 130 compute cores per chip (12×10 = 120 usable after harvest)
- 8 GDDR7 DRAM channels, 4GB each = 32 GB/device
- 1.5MB L1 per core
- Current mesh: (8,4) = TP=4, DP=8

## Current Performance (355B)
- P0a: 5.7 tok/s bs=1 (176ms ITL) — traced, CCL enabled
- P0b: 6.1 tok/s bs=1 (164ms ITL) — grid fix applied
- **P0c: ~7.1 tok/s bs=1** — reduce_scatter fix + warmup_batch=1 (PENDING verification)
- P1: batch>1 code deployed, NOT YET BENCHMARKED
- **GARBLED OUTPUT BUG FIXED**: reduce_scatter API change caused garbled output (Session 16)

## MANDATORY: Use a Team for ALL Work

**YOU MUST USE A TEAM. DO NOT WORK SOLO.**

Every session MUST create a team (e.g., `glm-bh-4`) with:
- **team-lead**: Coordinates, delegates, NEVER edits code or runs docker
- **implementer**: Edits code, runs rsync, manages containers, runs benchmarks
- **researcher** (optional): Background analysis via Gemini+Codex

**Team lead NEVER edits files, runs docker, or runs benchmarks.**
**Team lead NEVER edits files, runs docker, or runs benchmarks.**
**Team lead NEVER edits files, runs docker, or runs benchmarks.**

Delegate ALL implementation to the implementer. Read `plan/glm47_flash/_global/team-structure.md` for full rules.

## MANDATORY: Container restart=no

**NEVER use `restart: always` or `restart: unless-stopped` in docker compose.**
All containers MUST have `restart: no` (the default). Auto-restart after device crash
causes infinite crash loops that corrupt BH devices beyond IPMI recovery.

When starting containers, use `docker compose up -d` (creates fresh) not `docker restart`.

## Device Recovery
- Only IPMI power cycle works: `sudo ipmitool chassis power cycle`
- PCI FLR/SBR fail. tt-smi-metal -glx_reset CORRUPTS BH devices (WH-only).
- ~3.5 min recovery after IPMI cycle

## CRITICAL: Known Bugs / Do NOT Reintroduce
- **reduce_scatter API**: MoE STEP 7 MUST use `ttnn.experimental.reduce_scatter_minimal_async`
  with `num_links=rt.num_links, topology=rt.topology`. The standard `ttnn.reduce_scatter` produces
  GARBLED OUTPUT on BH Galaxy. Do NOT switch to the standard API.
- **warmup_batch**: Keep at 1 until batch>1 is verified. `max_batch_size` warmup crashes devices.
