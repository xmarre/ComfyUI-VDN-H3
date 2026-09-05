from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}\n--- old ---\n{old}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "vdn_h3/runtime.py",
    '_PREFETCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(\n'
    '    max_workers=1, thread_name_prefix="vdn-branch-prefetch")\n',
    '_PREFETCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(\n'
    '    max_workers=1, thread_name_prefix="vdn-branch-prefetch")\n'
    '_RECORD_STREAM_NEEDED = None\n\n\n'
    'def _record_stream_needed():\n'
    '    """Return whether explicit record_stream ownership is required.\n\n'
    '    PyTorch\'s cudaMallocAsync allocator is stream ordered already; record_stream\n'
    '    is a no-op there and current PyTorch warns when it is called on a tensor\n'
    '    created on the same stream. Cache the allocator decision for this process.\n'
    '    """\n'
    '    global _RECORD_STREAM_NEEDED\n'
    '    if _RECORD_STREAM_NEEDED is None:\n'
    '        try:\n'
    '            _RECORD_STREAM_NEEDED = torch.cuda.get_allocator_backend() != "cudaMallocAsync"\n'
    '        except Exception:\n'
    '            _RECORD_STREAM_NEEDED = True\n'
    '    return bool(_RECORD_STREAM_NEEDED)\n',
)
replace_once(
    "vdn_h3/runtime.py",
    '    @staticmethod\n'
    '    def _record_stream(tensor, stream):\n'
    '        seen = [tensor]\n',
    '    @staticmethod\n'
    '    def _record_stream(tensor, stream):\n'
    '        if not _record_stream_needed():\n'
    '            return\n'
    '        seen = [tensor]\n',
)

p = Path("tests/test_runtime_buffers.py")
text = p.read_text()
if "test_stream_prefetch_skips_record_stream_under_cuda_malloc_async" in text:
    raise SystemExit("runtime allocator test already present")
text += '''\n\ndef test_stream_prefetch_skips_record_stream_under_cuda_malloc_async(monkeypatch):\n    from vdn_h3 import runtime\n\n    class FakeTensor:\n        def __init__(self):\n            self.calls = 0\n\n        def record_stream(self, _stream):\n            self.calls += 1\n\n    fake = FakeTensor()\n    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")\n    monkeypatch.setattr(runtime, "_RECORD_STREAM_NEEDED", None)\n\n    runtime._StreamPrefetcher._record_stream(fake, object())\n\n    assert fake.calls == 0\n    assert runtime._RECORD_STREAM_NEEDED is False\n\n\ndef test_stream_prefetch_records_stream_for_non_async_allocator(monkeypatch):\n    from vdn_h3 import runtime\n\n    class FakeTensor:\n        def __init__(self):\n            self.calls = 0\n\n        def record_stream(self, _stream):\n            self.calls += 1\n\n    fake = FakeTensor()\n    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")\n    monkeypatch.setattr(runtime, "_RECORD_STREAM_NEEDED", None)\n\n    runtime._StreamPrefetcher._record_stream(fake, object())\n\n    assert fake.calls == 1\n    assert runtime._RECORD_STREAM_NEEDED is True\n'''
p.write_text(text)

replace_once(
    "pyproject.toml",
    'version = "1.3.1"\n',
    'version = "1.5.0"\n',
)
replace_once(
    "pyproject.toml",
    'authors = [{ name = "Saganaki22" }]\n',
    'authors = [{ name = "Saganaki22" }, { name = "xmarre" }]\n',
)
replace_once(
    "pyproject.toml",
    'Repository = "https://github.com/Saganaki22/ComfyUI-VDN-H3"\n'
    'Issues = "https://github.com/Saganaki22/ComfyUI-VDN-H3/issues"\n',
    'Repository = "https://github.com/xmarre/ComfyUI-VDN-H3"\n'
    'Issues = "https://github.com/xmarre/ComfyUI-VDN-H3/issues"\n',
)

replace_once(
    ".github/workflows/publish_action.yml",
    '  publish-node:\n'
    '    name: Publish Node\n'
    '    runs-on: ubuntu-latest\n',
    '  publish-node:\n'
    '    name: Publish Node\n'
    '    if: ${{ github.repository_owner == \'Saganaki22\' }}\n'
    '    runs-on: ubuntu-latest\n',
)

replace_once(
    "docs/UPSTREAM_RECONCILIATION.md",
    '- head: `fe6e6f2f26075f03dea09b5216e14db727af4b77`\n'
    '- prior snapshot already reconciled by this PR: `e49edae28266bcaa9b74988ac95ef4dd035f959c`\n'
    '- delta: 9 upstream commits\n',
    '- head: `b49130c26a70d12c542601c5bc4f7ee0f112ee2e`\n'
    '- prior snapshot already reconciled by this PR: `e49edae28266bcaa9b74988ac95ef4dd035f959c`\n'
    '- delta: 10 upstream commits\n',
)
replace_once(
    "docs/UPSTREAM_RECONCILIATION.md",
    '| `fe6e6f2f26075f03dea09b5216e14db727af4b77` | Scope compiler disable to VDN forwards and remove unload hook | Adopted and hardened. |\n',
    '| `fe6e6f2f26075f03dea09b5216e14db727af4b77` | Scope compiler disable to VDN forwards and remove unload hook | Adopted and hardened. |\n'
    '| `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` | Skip `record_stream` under `cudaMallocAsync` | Adopted in the fork-owned `vdn_h3.runtime` prefetcher. The fork moved lookahead ownership out of upstream `hybrid.py`, so the allocator guard is applied at the equivalent state-owned runtime boundary. |\n',
)
replace_once(
    "docs/UPSTREAM_RECONCILIATION.md",
    'Production GPU validation remains a separate gate and must still exercise the\n'
    'AIMDO-era Comfy build with VDN, Continuum, Spectrum and the Flow target-sparse\n'
    'external-sequence contract.\n',
    'The production GPU gate has now exercised the AIMDO-era Comfy build with VDN,\n'
    'Continuum, Spectrum, DiffAid, Untwisting RoPE, runtime LoRA bypass, streamed\n'
    'INT8-ConvRot branch weights, retained buffers, and Flow external-sequence API 2\n'
    'across multi-boundary mixed-grid continuation. The allocator guard additionally\n'
    'matches current upstream `b49130c...` and removes the observed no-op warning under\n'
    '`cudaMallocAsync`.\n',
)

Path("RELEASE_NOTES.md").write_text('''# ComfyUI-VDN-H3 v1.5.0\n\nv1.5.0 is the correctness/lifecycle release of the xmarre fork used by MiniMax H3 Flow-Aligned Regenerate mixed-grid Continuum. It reconciles upstream through current `b49130c26a70d12c542601c5bc4f7ee0f112ee2e`, retains the fork's stricter Comfy ownership model, and promotes external-sequence API 2 after real GPU multi-boundary validation.\n\n## Flow mixed-grid API 2\n\n- Keeps API 1 support for the target-sparse control path.\n- Adds `topology=mixed_grid_low_suffix` to `vdn_h3_external_sequence_v1` API 2.\n- Validates both the regular native low-carrier sequence and the actual mixed target-prefix/source-suffix row count.\n- Uses learned gated dense attention while disabling geometry-dependent local-window/linear-complement work only for the mixed external sequence.\n- Returns to ordinary VDN behavior for the fresh full target-grid stage.\n- Missing, stale or malformed external contracts fail closed.\n\n## Runtime LoRA lifecycle\n\n- Replaces the old forward-hook bypass chain with Comfy-managed weight/bias wrappers.\n- Keeps base weights unmaterialized in runtime bypass mode and bounds low-rank delta temporaries.\n- Supports released Stage-B/Turbo naming families and projection of full-width Turbo AdaLN LoRAs onto the native pruned/curve representation, including the required constant bias term.\n- Adapter ownership is isolated per Apply execution; clones remain equivalent and repeated execution does not accumulate adapters.\n\n## VDN resource ownership\n\n- `branch_weights=auto|stream|resident` uses effective free VRAM after accounting for an unloaded base model.\n- Quantized INT8-ConvRot branch weights remain streamed under managed-lifecycle policy.\n- Retained scan/window/activation scratch belongs to one `VDNState`; nested/concurrent execution falls back to isolated transient buffers.\n- One-block lookahead uses one bounded tensor-less executor and at most one state-owned outstanding result.\n- Completed prefetch results cannot be silently overwritten by a later request.\n- Flex BlockMask caching is bounded and keyed by full device/layout identity.\n\n## Current Comfy compiler and allocator compatibility\n\n- Adopts upstream v1.4.3's AIMDO incompatibility detection but scopes the process-global disable flag only around VDN diffusion-model forwards, with nested ownership and `finally` restoration.\n- Reconciles upstream `b49130c...`: when PyTorch uses `cudaMallocAsync`, VDN now skips `record_stream`, which is unnecessary for the stream-ordered allocator and produced a warning in the validated RTX Pro 6000 workflow. Native allocator paths retain explicit `record_stream`.\n\n## Real GPU validation\n\nThe integrated RTX Pro 6000 run exercised streamed INT8-ConvRot VDN branch weights, retained buffers, runtime adapter bypass, grouped attention, Spectrum + SA-Solver-PECE, DiffAid, Untwisting RoPE, learned H3 latent transfer, Flow API 2 mixed-grid continuation, fresh ordinary full-grid VDN stages, and multi-boundary Continuum completion. The Flow-side suffix DC bridge removed the remaining visible boundary flash while VDN stayed stable through the complete sequence.\n\nThis validates the interoperability/lifecycle path; it does not claim that every VDN branch/backend setting is numerically identical or universally faster.\n\n## Tests\n\n- Pinned Comfy + official OpenVDN oracle suite.\n- Current Comfy main import/node-registration/compiler-guard smoke.\n- Static/compile checks.\n- New allocator tests verify `cudaMallocAsync` skips `record_stream` while the native allocator path retains it.\n\n## Distribution\n\nThis GitHub release belongs to the xmarre correctness fork. The inherited Comfy Registry publisher identity remains `saganaki22`, so the fork release does **not** attempt to publish under that upstream registry identity.\n''')

Path(".github/workflows/release.yml").write_text('''name: release\n\non:\n  workflow_run:\n    workflows: [CI]\n    types: [completed]\n\npermissions:\n  contents: write\n\nconcurrency:\n  group: release-${{ github.repository }}-${{ github.event.workflow_run.head_branch }}\n  cancel-in-progress: false\n\njobs:\n  publish:\n    if: >-\n      github.repository_owner == 'xmarre' &&\n      github.event.workflow_run.conclusion == 'success' &&\n      github.event.workflow_run.event == 'push' &&\n      github.event.workflow_run.head_branch == 'main'\n    runs-on: ubuntu-latest\n    steps:\n      - name: Check out tested commit\n        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n        with:\n          persist-credentials: false\n          ref: ${{ github.event.workflow_run.head_sha }}\n\n      - name: Read package version\n        id: version\n        shell: bash\n        run: |\n          version="$(python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')"\n          if [[ ! "${version}" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]]; then\n            echo "Invalid project version: ${version}" >&2\n            exit 1\n          fi\n          echo "version=${version}" >> "${GITHUB_OUTPUT}"\n          echo "tag=v${version}" >> "${GITHUB_OUTPUT}"\n\n      - name: Refuse stale main workflow\n        env:\n          GH_TOKEN: ${{ github.token }}\n          RELEASE_SHA: ${{ github.event.workflow_run.head_sha }}\n        shell: bash\n        run: |\n          current_main="$(gh api "repos/${GITHUB_REPOSITORY}/commits/main" --jq '.sha')"\n          if [[ "${RELEASE_SHA}" != "${current_main}" ]]; then\n            echo "Tested commit ${RELEASE_SHA} is no longer current main (${current_main})." >&2\n            exit 1\n          fi\n\n      - name: Build release archive\n        env:\n          VERSION: ${{ steps.version.outputs.version }}\n        shell: bash\n        run: |\n          mkdir -p dist\n          archive="ComfyUI-VDN-H3-v${VERSION}.zip"\n          git archive --format=zip --prefix="ComfyUI-VDN-H3-v${VERSION}/" --output="dist/${archive}" HEAD\n          (cd dist && sha256sum "${archive}" > SHA256SUMS)\n\n      - name: Publish GitHub release\n        env:\n          GH_TOKEN: ${{ github.token }}\n          RELEASE_SHA: ${{ github.event.workflow_run.head_sha }}\n          TAG: ${{ steps.version.outputs.tag }}\n          VERSION: ${{ steps.version.outputs.version }}\n        shell: bash\n        run: |\n          if gh release view "${TAG}" >/dev/null 2>&1; then\n            echo "Release ${TAG} already exists; refusing to replace immutable release assets." >&2\n            exit 1\n          fi\n          gh release create "${TAG}" \\\n            "dist/ComfyUI-VDN-H3-v${VERSION}.zip" \\\n            dist/SHA256SUMS \\\n            --target "${RELEASE_SHA}" \\\n            --title "ComfyUI-VDN-H3 ${TAG}" \\\n            --notes-file RELEASE_NOTES.md\n''')
