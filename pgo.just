# pgo.just — shared PGO recipes, imported by per-crate justfiles.
# The importing justfile must define `pgo_dir`.
#
# Each flavor gets its own target dir (target-pgogen/, target-pgouse/): RUSTFLAGS
# are part of cargo's fingerprint, so sharing target/ between instrumented, plain,
# and profile-use builds forces a full rebuild on every flavor switch (~5 min each).
# With separate dirs each flavor rebuilds incrementally (~10 s for a leaf-crate
# change). Costs disk for the extra target dirs.
#
# Both target dirs can be pointed elsewhere. setup-bench.sh gives the baseline
# and working sides of an A/B their own pair, so each side keeps its own warm
# artifacts and neither invalidates the other.
pgogen_target_dir := env_var_or_default("PGO_GEN_TARGET_DIR", "target-pgogen")
pgouse_target_dir := env_var_or_default("PGO_USE_TARGET_DIR", "target-pgouse")

# Host triple. Passing an explicit --target (even the host's own triple) makes
# cargo stop applying RUSTFLAGS to host units (build scripts, proc macros):
# instrumentation never reaches them, and a profile-use flag change no longer
# invalidates their outputs, chiefly the DuckDB C++ cmake build, which no
# profile flag affects anyway.
host_target := `rustc -vV | sed -n 's/^host: //p'`

# Every recipe builds at the shipped instruction floor, so perf boxes measure
# exactly the codegen that releases ship. The floor is restated here because a
# set RUSTFLAGS REPLACES the .cargo/config.toml lists; keep the two in step
# (and with dispatch/src/cpu_features.rs). Empty on an arch without a defined
# floor; the recipes refuse to build then rather than silently dropping it.
floor_rustflags := if host_target =~ "x86_64" {
    "-Ctarget-cpu=x86-64-v3"
} else if host_target =~ "aarch64" {
    "-Ctarget-feature=+lse,+crc,+rdm,+dpb,+neon,+aes,+sha2,+dotprod,+ssbs,+rcpc,+bf16"
} else {
    ""
}

# Flags of every instrumented build. Counter updates are atomic: a plain update
# is a load, an add and a store, so workers running the same code at once
# overwrite each other's increments. The loss falls on the code all workers run
# concurrently and differs from one training run to the next, so the profile
# would under-weight the hottest paths by a varying amount.
pgogen_rustflags := "-Cprofile-generate=" + pgo_dir + " -Cllvm-args=-instrprof-atomic-counter-update-all"

# Run a cargo command instrumented for PGO profile generation, then merge the profiles.
#
#   just pgo-gen bench --bench clickbench
#   just pgo-gen test --test integration
alias pgo-generate := pgo-gen
pgo-gen *args:
    @[ -n "{{floor_rustflags}}" ] || { echo "no instruction floor defined for {{host_target}}" >&2; exit 1; }
    mkdir -p {{pgo_dir}}
    # PIVOT_SPIN_LIMIT=0 parks idle workers immediately during the profiled
    # workload, so the profile records the wait-heavy control-flow mix that
    # cold runs on the full dataset execute, deterministically rather than
    # as a timing-dependent draw per build.
    RUSTFLAGS="{{pgogen_rustflags}} {{floor_rustflags}}" \
        PIVOT_SPIN_LIMIT=0 \
        CARGO_TARGET_DIR={{pgogen_target_dir}} cargo {{args}}
    $(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata merge -o {{pgo_dir}}/merged.profdata {{pgo_dir}}

# Run a cargo command using the merged PGO profile.
#
#   just pgo-use bench --bench clickbench
pgo-use *args:
    @[ -n "{{floor_rustflags}}" ] || { echo "no instruction floor defined for {{host_target}}" >&2; exit 1; }
    RUSTFLAGS="-Cprofile-use={{pgo_dir}}/merged.profdata {{floor_rustflags}}" \
        CARGO_TARGET_DIR={{pgouse_target_dir}} cargo {{args}}

# Build instrumented for profile generation, without running or merging:
# for automation (A/B harnesses, the release build) that runs the workload
# itself, redirects the profiles via LLVM_PROFILE_FILE, and merges per side.
# Binaries land under target-pgogen/{{host_target}}/.
# lld because instrumentation can grow the text past the 128MB aarch64
# branch range, which GNU ld fails with relocation overflows.
pgo-gen-build subcmd *args:
    @[ -n "{{floor_rustflags}}" ] || { echo "no instruction floor defined for {{host_target}}" >&2; exit 1; }
    mkdir -p {{pgo_dir}}
    RUSTFLAGS="{{pgogen_rustflags}} {{floor_rustflags}} -Clink-arg=-fuse-ld=lld" \
        CARGO_TARGET_DIR={{pgogen_target_dir}} cargo {{subcmd}} --target {{host_target}} {{args}}

# Like pgo-use but with a caller-supplied profdata path. rustc records the
# profile file in dep-info, so even at a constant path a regenerated profile
# rebuilds every Rust unit; passing a per-profile path makes that rebuild a
# consequence of the flags hash too, immune to mtime comparisons against
# artifacts restored from another machine. Build-script outputs (unflagged
# thanks to the explicit --target) stay cached either way. Binaries land
# under target-pgouse/{{host_target}}/.
# The path is absolutized because cargo invokes rustc from each crate's own
# directory, so a relative one resolves only for workspace crates.
pgo-use-with profdata subcmd *args:
    @[ -n "{{floor_rustflags}}" ] || { echo "no instruction floor defined for {{host_target}}" >&2; exit 1; }
    @[ -f "{{profdata}}" ] || { echo "profile '{{profdata}}' does not exist" >&2; exit 1; }
    RUSTFLAGS="-Cprofile-use=$(readlink -f "{{profdata}}") {{floor_rustflags}}" \
        CARGO_TARGET_DIR={{pgouse_target_dir}} cargo {{subcmd}} --target {{host_target}} {{args}}

# Merge a training run's raw profiles into the indexed file -Cprofile-use
# reads. Fails loudly when the run produced none, so a build can never
# quietly proceed un-PGOed.
pgo-merge out dir:
    #!/usr/bin/env bash
    set -euo pipefail
    raw_count="$(find "{{dir}}" -name '*.profraw' | wc -l)"
    [[ "$raw_count" -gt 0 ]] || { echo "no .profraw files under {{dir}}; the training run produced no profile" >&2; exit 1; }
    "$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata" merge -o "{{out}}" "{{dir}}"/*.profraw
    echo "merged $raw_count profraw files into {{out}}"

# Tripwire for the profile applying at all: the decode family's
# monomorphizations in the built binary must appear in the profile it was
# compiled against. Zero overlap means the binary is effectively un-PGOed,
# which is silent at compile time and shows up only as a mystery regression.
# Whole mangled names, so the check holds under both legacy and v0 mangling.
pgo-verify binary profdata:
    #!/usr/bin/env bash
    set -euo pipefail
    # Absolutized so callers can pass paths relative to their own working
    # directory (recipes run from the justfile's directory).
    binary="$(readlink -f "{{binary}}")"
    profdata="$(readlink -f "{{profdata}}")"
    family="RleDecoder"
    # The greps legitimately match nothing when the family is fully inlined;
    # the zero-overlap case below is the loud failure.
    binary_symbols="$(nm "$binary" | awk '{print $NF}' | grep "$family" | sort -u || true)"
    if [[ -z "$binary_symbols" ]]; then
        echo "warning: no $family symbols visible to nm, skipping the profile-overlap check" >&2
        exit 0
    fi
    profile_symbols="$("$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata" \
        show -all-functions "$profdata" 2>/dev/null \
        | grep -oE '[[:alnum:]_.$]*'"$family"'[[:alnum:]_.$]*' | sort -u || true)"
    covered="$(comm -12 <(printf '%s\n' "$binary_symbols") <(printf '%s\n' "$profile_symbols") | wc -l)"
    [[ "$covered" -gt 0 ]] || { echo "the binary shares no $family symbols with its profile; the profile did not apply" >&2; exit 1; }
    echo "verified: $covered $family monomorphizations carry profile records"

# Remove all PGO profiles and merged data.
pgo-clean:
    rm -rf {{pgo_dir}}
