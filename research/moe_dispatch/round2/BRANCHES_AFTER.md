# Compiler branches after round2 integration

Date: 2026-10-07. Existing branch: `research/moe-supply-first-v3`.

Pre-integration HEAD: `480d558c72f8fce19572b2408eb02a1ae52c40eb`. Inventory HEAD before final validation/document commit: `272b20fbad791808f6e4f9a5a913c3df81695f7a`.

**All 33 unique owned candidate heads are ancestors of the integrated HEAD.** 23 heads were joined by explicit `--no-ff` merge commits; remaining heads became ancestors through those merges. No local branch or worktree was created/deleted/rewritten. Canonical dirty worktrees were not edited.

| Candidate | SHA | Result | Merge commit |
|---|---|---|---|
| `refs/heads/research/projection-pipeline-20260928` | `32dc43bdd7c9ee52b4eb1315d93882d0cdf77136` | merged_with_conflicts | `99fca28c1d428e731915a47512912bc8cbefdb2d` |
| `refs/heads/feat/routed-moe-compiler-substrate` | `54a68e6969651eb5c1c3397ac59bed6585c08806` | merged_with_conflicts | `86db490ac4dc028cc23c13254ce51fe638ec9d72` |
| `refs/heads/codex/moe-e2e-sync` | `1add2606985e5962df3a0e7c2ea90b8b302a2ec8` | merged_with_conflicts | `d6ec16179ddc594efe6eeffdd3c4d196d53747b8` |
| `refs/heads/fix/portable-developer-paths` | `29fdbc17a8877d44bf54ad58bc2d538449a263c3` | merged | `ad2b20a79591d3add54dee79d6f3e582d4fadaaf` |
| `refs/heads/agent/portable-compiler-tooling` | `1a26215f831c38e1bd92e83eab7e87f2a4ef47d0` | merged | `8e5957d8e189907243ee3394452f092d8b238a86` |
| `refs/heads/local/archive-e2e-compiler-expert-20260727` | `ea9e8c7a1f4ae3a06ee8896ceaf3453977179840` | merged_with_conflicts | `ccfc7c29b51122ae195b5eb13c2373bbe1c3a867` |
| `refs/heads/local/archive-old-topk-compiler-20260727` | `1468974f29828a7dea898943ca5fffaff4976ad4` | merged_with_conflicts | `3cc24ec08efe77dae7cea70dc2d177da28602aea` |
| `refs/heads/local/candidate-compiler-topk-contract-20260726` | `71ba7ff4f1d697d5a444f85da6bc8e4484090772` | merged_with_conflicts | `d045bdf5709819ade13d0d3561497fb6c6b6813e` |
| `refs/heads/local/compiler-expert-ffn-20260727` | `8cfb0d071524d18c5a3bc98d2ef498509e123389` | merged_with_conflicts | `597787fe787dd1766343e3167f86c8b43f82ab68` |
| `refs/heads/feat/nemotron3-mamba2-system` | `dc6ea46db2ed2287a57d233bec3cf14dfae1ffb9` | merged | `7b26994fa88232b8873ca63f3ce7fbdc6d70a32c` |
| `refs/remotes/origin/moe/shared-expert-preview-20260814` | `5b06a5927f1ccd04a2dbe6a8d9caec640b49e25a` | merged_with_conflicts | `f9497f53381666ba9a594d6c5697becc35104083` |
| `refs/heads/feat/nemotron3-mamba-compiler` | `0aabc6384aef661bef46f1e3c58997bd276e2ce9` | merged_with_conflicts | `567e37333c75e617253be4546cacd9f9301c83f7` |
| `refs/heads/review/mamba-kda-c4-kimi` | `3b4e6faf5e2f7de8e7935a0b74256261b749d0cc` | merged_with_conflicts | `d42deb97b987baf90d2026ebe3c18cc851c47128` |
| `refs/heads/review/mamba-kda-c1-state-isa` | `62a34b6777f17e475f561657742865f2e7ca84f2` | already_ancestor | `covered by earlier merge` |
| `refs/heads/review/mamba-kda-c2-plena-substrate` | `8950eaed5dbfc119b72ea0cdbd19086d93ffbd8f` | already_ancestor | `covered by earlier merge` |
| `refs/heads/review/mamba-kda-c3-nemotron` | `975f9a5239c5b0ad9dc6b9ac7fbcc8c78b2c1e6a` | already_ancestor | `covered by earlier merge` |
| `refs/heads/feature/mamba-kda-support` | `8bc274b52b2bcee79d8aa3a538037062a277a2d1` | merged_with_conflicts | `b4975cacaacf0f4edd2443da1c5838ca639f0201` |
| `refs/heads/feat/static-kda-gpu-evidence` | `be490d20549b4ef1b05c3b5aa9ec6e23e7a88307` | already_ancestor | `covered by earlier merge` |
| `refs/heads/feat/static-kda-official-layer` | `d0d71aee1a256d912b2581981015840f516d5b1a` | already_ancestor | `covered by earlier merge` |
| `refs/heads/local/shared-deps-20260810` | `cabe01f59f17d4caa9c688cabf0d6e3b7850c3e8` | merged_with_conflicts | `6c55c20e3ee05e0ff3b858114615860e0734aa41` |
| `refs/remotes/origin/feat/fixed-route-expert-grouping` | `e4cd6c5f9887f0118a43e3e5484130f45291fce9` | merged_with_conflicts | `d2742a43d888a15d41b499b23383b315b1a78833` |
| `refs/heads/feat/matrix-sram-lcompute` | `9fd6a59299fd9432e455f674f1bf9d414628ec36` | already_ancestor | `covered by earlier merge` |
| `refs/heads/archive/pr-79-before-scope-cleanup-20260905` | `1955abba267e4c9d9874221f53a9c16db984c8cf` | merged_with_conflicts | `e8600519ecb985ac9eaa82bca054a3e004b7347d` |
| `refs/heads/archive/compiler-lcompute-full-671dc37` | `671dc37cc12938728276569bfaebaf52276c7631` | merged_with_conflicts | `34ca47f23b7fd8d9ca9ba6d4db3694f0c8ab1d14` |
| `refs/remotes/origin/review/moe-normal-export-20260905` | `a6730b5d6cfd3190d571236bb02a4a2babaaabba` | merged | `665dd50b4aa1916fdf67cd3aa8d82d479af6662f` |
| `refs/heads/review/matrix-lcompute-20260905` | `9637858ad9b64ac8c02adcc0ec0b1d749ae6a6fa` | merged_with_conflicts | `ae5bc956a4d5910db1bfda6a9dd4b9ec55f6a012` |
| `refs/heads/research/ltile-architecture-20260914` | `bf9687d693a2d2078580ffc80d4d703c80bd4801` | already_ancestor | `covered by earlier merge` |
| `refs/heads/research/ltile-full-layer` | `8a040dc5760fe6ce8d57aa1ad62e108c4f80bcce` | already_ancestor | `covered by earlier merge` |
| `refs/heads/research/ltile-layer-service` | `a05af53dd57683aa308c05dc9b4a8ace4ee4edba` | already_ancestor | `covered by earlier merge` |
| `refs/remotes/origin/research/projection-pipeline-20260928` | `0b2f549ac9cebeb3f328b65fc8ea6f36e1c1be49` | already_ancestor | `covered by earlier merge` |
| `refs/round2/live/agent/portable-compiler-tooling` | `cf04518cd8c482e48ade57e23b7926c2b26244c8` | merged | `294990a0c0eb7fc3709980995a710bc397a19be8` |
| `refs/round2/live/feat/routed-moe-compiler-substrate` | `e272404f74d3e25bc882638fec0d41919c10e76b` | merged_with_conflicts | `ca499986ce893af2101335452671f8aa1c092537` |
| `refs/round2/live/review/moe-normal-export-20260905` | `c45cf7b069c9f16cfe2d5a2a454ee4e78d35f0be` | merged | `272b20fbad791808f6e4f9a5a913c3df81695f7a` |

## Live remote verification

All 36 live Compiler heads were fetched into the non-head namespace `refs/round2/live/*`. Live-only mcl123 work was additionally integrated: portable tooling through `cf04518`, routed substrate through `e272404`, and the live MoE normal-export head `c45cf7`. Subsequent collaborator-only tips were left out. In particular, IPD tip `c3ea4e8` is authored by Qichao (Arlo) Wang and contains no mcl123 commit missing from integrated history. This is ownership-based scope, not removal of already inherited collaborator history.

## Preserved versions and executable profiles

Conflict stages and user worktree patches are in `research/moe_dispatch/archive/`. The Matrix/L-TILE + route profile is authoritative. Older X_MAMBA, X_STATE, L_SCATTER_M and consolidated-C_ROUTE prototype encodings are retained as archived profiles and history; they are not simultaneously executable through conflicting opcodes. Archive test files are excluded from ordinary recursive collection. Individual retired opcode integration tests report explicit skips; codec-only historical tests remain executable. See `ISA_PROFILES.md`.
