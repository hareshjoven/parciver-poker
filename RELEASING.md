# Releasing the blueprint tables

`parciver play nlhe` needs the v2-8B blueprint (1.27 GB) and three card-bucket tables (277 MB). They are too large
for git, so they are published as assets of a GitHub release and fetched by `parciver fetch-blueprint`.

## What is published

| file in the release | installed as | uncompressed |
|---|---|---|
| `blueprint.bin.gz` | `~/.parciver/blueprint.bin` | 1,271,160,285 bytes |
| `abstraction_flop_buckets.npy.gz` | `~/.parciver/abstraction/flop_buckets.npy` | 2,573,712 bytes |
| `abstraction_turn_buckets.npy.gz` | `~/.parciver/abstraction/turn_buckets.npy` | 27,920,228 bytes |
| `abstraction_river_buckets.npy.gz` | `~/.parciver/abstraction/river_buckets.npy` | 246,312,636 bytes |

`src/parciver/assets.json` is the manifest: for every file, the SHA-256 and size of the **uncompressed** bytes. The
downloader decompresses while streaming, hashes as it goes, and moves a file into place only if both match;
otherwise it deletes the partial file and installs nothing. The files are plain numeric tables read by the Rust
engine; nothing downloaded is executed.

The blueprint is the `v2-8B` strategy: 8,000,247,000 MCCFR iterations at 200 big blinds, the one the SLUMBOT-2
match in [RESULTS.md](RESULTS.md) was played with (the first match used a 2B-iteration blueprint); sha256 `fd27fe50...ae071`.

## Procedure (maintainer)

1. Build the assets from the local training outputs (this is the only writer of `assets.json`):

   ```bash
   python scripts/make_release_assets.py --out ./release-assets
   ```

2. Create a release tagged `blueprint-v1` and attach the four `.gz` files from `./release-assets`.
3. Set `"base_url"` in `src/parciver/assets.json` to
   `https://github.com/<owner>/<repo>/releases/download/blueprint-v1` and commit.
4. From a clean machine: `pip install .`, `parciver fetch-blueprint`, `parciver play nlhe --hands 1`.

Users can also point at a mirror with `parciver fetch-blueprint --url <directory>` or `$PARCIVER_ASSET_URL`.

# Releasing the flop value nets

The full GPU configuration (README, tier 3) also needs a flop value net. Two are published, as a second release
tagged `flopnets-v1`, fetched by `parciver fetch-nets` and listed under `"nets"` in the same manifest:

| file in the release | installed as | bytes | sha256 |
|---|---|---|---|
| `flopnet_N0.pt` | `~/.parciver/nets/N0.pt` | 111,059,281 | `a8bf00be...38ef65` |
| `flopnet_N1.pt` | `~/.parciver/nets/N1.pt` | 111,059,345 | `54bae749...f97d72` |

**N0** is the net the SLUMBOT-2 match in [RESULTS.md](RESULTS.md) was played with (flop search, 500 iterations,
suit-symmetric leaves). **N1** is the net the live configuration uses since D2b, and the default of
`scripts/slumbot_run.py`; `--net N0` plays the match's net. Both are 2,048 wide and 5 deep, about 111 MB. Releasing
the nets changes the setup, not any result: no number in RESULTS.md moves.

A net file holds tensors and plain numbers only, and `value_net.load` reads it with `torch.load(weights_only=True)`,
which refuses anything else: loading a downloaded net executes no code. The release script loads each net that way
before it will pack it, and refuses a file whose SHA-256 is not the one the records pin.

Procedure: `python scripts/make_release_assets.py --out ./release-assets --only nets` (it rewrites only the `"nets"`
entry of `assets.json`), create a release tagged `flopnets-v1` with the two `.pt` files, then from a clean machine
run `parciver fetch-nets`. A live Slumbot run additionally needs a registered mock run on the chosen net, committed
as `records/livesearch/mock_check.json`. The mock run (`--mock`) needs the nets, the blueprint tables and Slumbot's
`sample_api.py` (not in this repository; see the README).
