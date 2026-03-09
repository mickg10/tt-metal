# GLM-4.7-Flash on Blackhole Galaxy — tt-metal workspace

## CRITICAL: Remote Machine Rules

This workspace is the LOCAL source of truth. The Blackhole Galaxy machine
(`ssh -p 55212 mick@38.97.6.6`) is a SLAVE that receives code via rsync.

### FORBIDDEN on the remote BH machine:
- `git clone`, `git checkout`, `git pull`, `git fetch`, `git rebase`, `git commit`
- `git switch`, `git branch`, `git merge`, `git stash`, `git reset`
- ANY git command whatsoever — the remote has NO .git directory

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

### Container commands (run ON REMOTE via SSH):
```bash
ssh -p 55212 mick@38.97.6.6
cd /home/mick/ws/glm47_flash_blackhole_galaxy/docker_tt
docker compose -p bh2 --env-file dev/.env.glm47_blackhole \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  restart vllm-tt
```

### Benchmark commands (run ON REMOTE via SSH):
```bash
# Inside the container or via docker exec:
python tests/bench_decode.py --url http://localhost:8088 --gen-tokens 500 \
  --only-batch 1 --skip-combined --prefill-contexts 0
```

## Key Model Code
- `models/demos/glm4_moe_lite/tt/moe_tt.py` — MoE routing, sparse matmul config (GRID BUG here)
- `models/demos/glm4_moe_lite/tt/decoder_layer_tt.py` — Forward pass
- `models/demos/glm4_moe_lite/tt/layer_weights.py` — Weight loading
- `models/demos/glm4_moe_lite/tt/generator_vllm.py` — vLLM interface

## Blackhole Hardware
- 13×10 = 130 compute cores per chip
- 8 GDDR7 DRAM channels, 4GB each
- 1.5MB L1 per core
- Current mesh: (8,4) = TP=4, DP=8
