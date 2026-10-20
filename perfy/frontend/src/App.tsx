import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  CATEGORY_META,
  CATEGORY_ORDER,
  type AnnotateLine,
  type AnnotateResponse,
  type Category,
  type FlameResponse,
  type Meta,
  type PipelineSummary,
  type StatResponse,
  type TracksResponse,
} from "./api";
import { SummaryView } from "./components/SummaryView";
import { buildCategories, Timeline } from "./components/Timeline";
import { FlameGraph } from "./components/Flamegraph";
import {
  CodePanel,
  ModeToggle,
  type CyclesMode,
  type MemoryMode,
} from "./components/CodePanel";
import { Splitter } from "./components/Splitter";
import { InstructionDetail } from "./components/InstructionDetail";
import { Spinner } from "./components/Spinner";

/// Number of buckets per category (and per-CPU sub-bucket) the timeline
/// is rendered with. Samply uses ~2000+ for sub-millisecond detail; at
/// ~1500px of pane width that gives ~1 bin per pixel and roughly matches
/// what you'd expect from a per-event waveform without losing sharp
/// dips/peaks. JSON payload at this density is on the order of a few MB
/// for a 16-cpu profile, which is comfortably fast.
const TIME_BINS = 2000;

/// The Top-Down Summary tab is wired up end-to-end (backend computation,
/// frontend rendering, zoom integration) but currently disabled because
/// the AMD Zen 4 multiplexing constraints under `cycles:D` keep backend/
/// bad-spec metrics unavailable in the workflow we actually use. Flip
/// this back to `true` once the recording pipeline is sorted out — no
/// other code change is required.
const SUMMARY_TAB_ENABLED = false;

function formatZoomSpan(span: number, totalMs: number): string {
  const ms = (span / TIME_BINS) * totalMs;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms.toFixed(0)}ms`;
}

function Toolbar({
  meta,
  zoomStack,
  goToLevel,
  visibleCats,
  setVisibleCats,
}: {
  meta: Meta;
  zoomStack: [number, number][];
  goToLevel: (idx: number) => void;
  visibleCats: Set<Category>;
  setVisibleCats: (s: Set<Category>) => void;
}) {
  const ms = meta.duration_ns / 1e6;
  const availCats = CATEGORY_ORDER.filter((c) => meta.categories.includes(c));
  const toggleCat = (c: Category) => {
    const next = new Set(visibleCats);
    if (next.has(c)) next.delete(c);
    else next.add(c);
    setVisibleCats(next);
  };
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "3px 10px",
        borderBottom: "1px solid oklch(0.93 0.005 250)",
        background: "white",
        height: 26,
      }}
    >
      {/* Zoom breadcrumbs — each prior zoom level is a pill the user
          can click to pop back to. The leftmost pill is always the
          full recording; the rightmost (current) is bold and inert.
          Replaces the single "Reset zoom" button: you can now step
          back one level instead of jumping all the way out. */}
      <div
        style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}
      >
        {zoomStack.map((r, idx) => {
          const isCurrent = idx === zoomStack.length - 1;
          const span = r[1] - r[0];
          const label =
            idx === 0
              ? "Full"
              : formatZoomSpan(span, ms);
          return (
            <span
              key={idx}
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              <button
                onClick={isCurrent ? undefined : () => goToLevel(idx)}
                disabled={isCurrent}
                title={
                  isCurrent
                    ? "Current zoom"
                    : `Back to ${label}`
                }
                style={{
                  padding: "2px 8px",
                  borderRadius: 4,
                  border: "1px solid oklch(0.92 0.005 250)",
                  background: isCurrent ? "oklch(0.95 0.02 250)" : "white",
                  color: isCurrent
                    ? "oklch(0.2 0.01 250)"
                    : "oklch(0.4 0.12 250)",
                  fontSize: 11,
                  fontWeight: isCurrent ? 600 : 400,
                  cursor: isCurrent ? "default" : "pointer",
                  fontFamily: "inherit",
                  height: 20,
                  lineHeight: 1,
                }}
              >
                {label}
              </button>
              {!isCurrent && (
                <span style={{ color: "oklch(0.7 0.01 250)" }}>›</span>
              )}
            </span>
          );
        })}
      </div>
      {/* Per-category checkboxes — toggle whether that track shows up
          in the timeline. Off by default = unchecked. Color of the dot
          mirrors the track tint so the user can match label↔track. */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {availCats.map((c) => {
          const on = visibleCats.has(c);
          const cm = CATEGORY_META[c];
          return (
            <label
              key={c}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                fontSize: 11,
                color: on ? "oklch(0.25 0.01 250)" : "oklch(0.6 0.01 250)",
                cursor: "pointer",
                userSelect: "none",
              }}
            >
              <input
                type="checkbox"
                checked={on}
                onChange={() => toggleCat(c)}
                style={{ margin: 0, cursor: "pointer" }}
              />
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 2,
                  background: `oklch(0.78 0.18 ${cm.hue})`,
                  display: "inline-block",
                  opacity: on ? 1 : 0.3,
                }}
              />
              {cm.short.toLowerCase()}
            </label>
          );
        })}
      </div>
      <div style={{ flex: 1 }} />
      <div
        style={{
          fontSize: 11,
          color: "oklch(0.55 0.01 250)",
          fontFamily: "ui-monospace, monospace",
        }}
      >
        {meta.perf_data.split("/").pop()} · {(ms / 1000).toFixed(2)} s ·{" "}
        {meta.cpus.length} cpus
      </div>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [consolidated, setConsolidated] = useState<TracksResponse | null>(null);
  const [perCpu, setPerCpu] = useState<TracksResponse | null>(null);
  /// Optional perf-stat-derived tracks (memory throughput + L3
  /// read-miss latency). null while loading; an `available: false`
  /// response means the recording wasn't taken with stat.csv and the
  /// frontend should hide the memory rows entirely.
  const [stat, setStat] = useState<StatResponse | null>(null);
  const [pipelineSummary, setPipelineSummary] = useState<PipelineSummary | null>(null);
  /// Which tab the middle pane shows: the auto-fetched "summary"
  /// (pipeline top-down metrics) or the on-demand "flame" graph.
  const [middleTab, setMiddleTab] = useState<"summary" | "flame">(
    SUMMARY_TAB_ENABLED ? "summary" : "flame",
  );

  /// Zoom history as a stack of bin-space ranges. The bottom entry is
  /// always the full recording; pushing onto it records a new zoom-in
  /// level, and the breadcrumb pills in the toolbar let the user pop
  /// back to any prior level (including the full view). `range` below
  /// is just the top of this stack.
  const [zoomStack, setZoomStack] = useState<[number, number][]>([
    [0, TIME_BINS - 1],
  ]);
  const range = zoomStack[zoomStack.length - 1];
  const pushRange = useCallback((r: [number, number]) => {
    setZoomStack((s) => {
      const cur = s[s.length - 1];
      // No-op pushes (e.g. user drags a marquee that exactly matches
      // the current view) would clutter the breadcrumb without
      // changing anything — drop them.
      if (cur && cur[0] === r[0] && cur[1] === r[1]) return s;
      return [...s, r];
    });
  }, []);
  const goToLevel = useCallback((idx: number) => {
    setZoomStack((s) =>
      idx >= s.length - 1 ? s : s.slice(0, Math.max(idx + 1, 1)),
    );
  }, []);
  const resetZoom = useCallback(() => goToLevel(0), [goToLevel]);
  const [expandedCats, setExpandedCats] = useState<Set<Category>>(new Set());
  const [selectedCat, setSelectedCat] = useState<Category | null>(null);
  /// Which categories the user wants visible in the timeline. Defaults
  /// to every category present in the recording. Toggling these only
  /// hides timeline rows — the flamegraph keeps whatever the user
  /// last selected, even if its category gets unchecked here.
  const [visibleCats, setVisibleCats] = useState<Set<Category>>(new Set());
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const [cyclesMode, setCyclesMode] = useState<CyclesMode>("relative");
  const [memoryMode, setMemoryMode] = useState<MemoryMode>("relative");
  const [activeAsm, setActiveAsm] = useState<AnnotateLine | null>(null);

  const [flame, setFlame] = useState<FlameResponse | null>(null);
  const [flameLoading, setFlameLoading] = useState(false);

  const [annotation, setAnnotation] = useState<AnnotateResponse | null>(null);
  const [annotationLoading, setAnnotationLoading] = useState(false);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  // Resizable pane heights — only consulted when the corresponding pane is
  // actually visible. The min ensures content remains readable; the max is
  // unenforced (we let the splitter clamp via window.innerHeight).
  const [timelineH, setTimelineH] = useState(360);
  const [flameH, setFlameH] = useState(280);
  // Splitter clamps: lower bound 0 so the user can drag a pane all the
  // way closed (giving the panes below it the full screen). Upper
  // bound stays generous so a single pane can grow nearly full
  // viewport too.
  const onResizeTimeline = useCallback((dy: number) => {
    setTimelineH((h) =>
      Math.max(0, Math.min(window.innerHeight - 40, h + dy)),
    );
  }, []);
  const onResizeFlame = useCallback((dy: number) => {
    setFlameH((h) =>
      Math.max(0, Math.min(window.innerHeight - 40, h + dy)),
    );
  }, []);

  // Bootstrap
  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(String(e)));
  }, []);

  // Load tracks (both consolidated + per-cpu) once we have meta
  useEffect(() => {
    if (!meta) return;
    const cats = CATEGORY_ORDER.filter((c) => meta.categories.includes(c));
    if (cats.length === 0) return;
    let cancelled = false;
    Promise.all([
      api.tracks(cats, true, TIME_BINS),
      api.tracks(cats, false, TIME_BINS),
      api.stat(TIME_BINS),
    ])
      .then(([cons, per, st]) => {
        if (cancelled) return;
        setConsolidated(cons);
        setPerCpu(per);
        setStat(st);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [meta]);

  /// Pipeline summary follows the timeline zoom: re-fetch whenever
  /// the user pans/zooms the time range so the Top-Down breakdown
  /// reflects what's actually on screen. AbortController lets a fast
  /// drag cancel in-flight requests so we don't end up rendering a
  /// stale summary.
  const summaryAbort = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!SUMMARY_TAB_ENABLED) return;
    if (!meta) return;
    const span = TIME_BINS;
    const tFull = meta.duration_ns;
    const lo = meta.time_start_ns + Math.round((range[0] / span) * tFull);
    const hi = meta.time_start_ns + Math.round(((range[1] + 1) / span) * tFull);
    const isFull = range[0] === 0 && range[1] === span - 1;
    summaryAbort.current?.abort();
    const ac = new AbortController();
    summaryAbort.current = ac;
    api
      .pipelineSummary(isFull ? undefined : { lo_ns: lo, hi_ns: hi })
      .then((r) => {
        if (!ac.signal.aborted) setPipelineSummary(r);
      })
      .catch((e) => !ac.signal.aborted && setError(String(e)));
    return () => ac.abort();
  }, [meta, range]);

  const allCats = useMemo(() => {
    if (!consolidated || !perCpu) return [];
    return buildCategories(consolidated, perCpu);
  }, [consolidated, perCpu]);

  /// Initialise visibleCats once the recording's categories are known.
  /// Default = everything present.
  useEffect(() => {
    if (!meta) return;
    setVisibleCats((prev) => {
      if (prev.size > 0) return prev;
      return new Set(meta.categories);
    });
  }, [meta]);

  /// Auto-show the flamegraph: pick the first available category
  /// (preferring "cycles") as soon as we have data, so the user lands
  /// in a useful view without an extra click.
  useEffect(() => {
    if (selectedCat || !meta || allCats.length === 0) return;
    const preferred: Category = (
      meta.categories.includes("cycles") ? "cycles" : meta.categories[0]
    ) as Category;
    setSelectedCat(preferred);
  }, [meta, allCats, selectedCat]);

  const cats = useMemo(
    () => allCats.filter((c) => visibleCats.has(c.id)),
    [allCats, visibleCats],
  );

  // When a category is selected, fetch its flamegraph for the current zoom range.
  const flameAbort = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!selectedCat || !meta || !consolidated) {
      setFlame(null);
      return;
    }
    const span = TIME_BINS;
    const t0 = meta.time_start_ns;
    const tFull = meta.duration_ns;
    const lo = t0 + Math.round((range[0] / span) * tFull);
    const hi = t0 + Math.round(((range[1] + 1) / span) * tFull);
    const isFull = range[0] === 0 && range[1] === span - 1;
    flameAbort.current?.abort();
    const ac = new AbortController();
    flameAbort.current = ac;
    setFlameLoading(true);
    api
      .flamegraph(`cat:${selectedCat}`, isFull ? undefined : { lo_ns: lo, hi_ns: hi })
      .then((r) => {
        if (!ac.signal.aborted) setFlame(r);
      })
      .catch((e) => !ac.signal.aborted && setError(String(e)))
      .finally(() => !ac.signal.aborted && setFlameLoading(false));
    return () => ac.abort();
  }, [selectedCat, range, meta, consolidated]);

  const onSelectFrame = useCallback((label: string) => {
    if (!label || label === "(all)" || label === "[unknown]") return;
    setSelectedFrame(label);
    setAnnotationError(null);
    setAnnotationLoading(true);
    api
      .annotate(label)
      .then(setAnnotation)
      .catch((e) => setAnnotationError(String(e)))
      .finally(() => setAnnotationLoading(false));
  }, []);

  const toggleExpand = useCallback((cat: Category) => {
    setExpandedCats((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }, []);

  if (error) {
    return (
      <div style={{ padding: 16, fontFamily: "Inter, sans-serif" }}>
        <div
          style={{
            color: "oklch(0.4 0.18 25)",
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          Error
        </div>
        <pre style={{ marginTop: 8, fontFamily: "ui-monospace, monospace" }}>
          {error}
        </pre>
      </div>
    );
  }
  if (!meta) {
    return (
      <div style={{ height: "100vh" }}>
        <Spinner size={36} label="Loading profile…" />
      </div>
    );
  }

  const ms = meta.duration_ns / 1e6;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
        fontFamily:
          '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
        color: "oklch(0.2 0.01 250)",
        background: "oklch(0.99 0.003 250)",
        position: "relative",
      }}
    >
      <Toolbar
        meta={meta}
        zoomStack={zoomStack}
        goToLevel={goToLevel}
        visibleCats={visibleCats}
        setVisibleCats={setVisibleCats}
      />

      {/* Tracks pane — always fixed-height now so the splitter below
          actually has somewhere to push into. The splitter then sits
          between tracks and the flamegraph; drag it down to grow the
          tracks pane (lets you see all per-cpu rows at once). */}
      <div
        style={{
          height: timelineH,
          flex: "0 0 auto",
          overflow: "auto",
          borderBottom: "1px solid oklch(0.92 0.005 250)",
        }}
      >
        {allCats.length === 0 ? (
          <Spinner label="Loading tracks…" />
        ) : (
          <Timeline
            cpus={meta.cpus}
            cats={cats}
            timeBins={TIME_BINS}
            durationMs={ms}
            selectedCat={selectedCat}
            setSelectedCat={setSelectedCat}
            expandedCats={expandedCats}
            toggleExpand={toggleExpand}
            range={range}
            pushRange={pushRange}
            resetZoom={resetZoom}
            stat={stat}
          />
        )}
      </div>

      <Splitter onResize={onResizeTimeline} />

      {selectedCat && (
        <div
          style={{
            height: selectedFrame ? flameH : undefined,
            flex: selectedFrame ? "0 0 auto" : 1,
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
            background: "white",
          }}
        >
          {/* Tab strip — Summary | Flame graph. Same place where the
              flamegraph used to live; the two panes share this slot.
              Hidden entirely when the Summary tab is disabled, so the
              flamegraph fills the whole middle pane with no dead chrome. */}
          {SUMMARY_TAB_ENABLED && <div
            style={{
              display: "flex",
              alignItems: "stretch",
              borderBottom: "1px solid oklch(0.92 0.005 250)",
              background: "oklch(0.985 0.003 250)",
              fontSize: 12,
              flex: "0 0 auto",
            }}
          >
            {(
              [
                { id: "summary", label: "Summary" },
                { id: "flame", label: "Flame graph" },
              ] as const
            ).map((t) => {
              const active = middleTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setMiddleTab(t.id)}
                  style={{
                    border: "none",
                    background: "transparent",
                    padding: "10px 18px",
                    cursor: "pointer",
                    fontSize: 13,
                    fontWeight: active ? 600 : 400,
                    color: active
                      ? "oklch(0.25 0.16 250)"
                      : "oklch(0.5 0.01 250)",
                    borderBottom: active
                      ? "2px solid oklch(0.55 0.18 250)"
                      : "2px solid transparent",
                    marginBottom: -1,
                    fontFamily: "inherit",
                  }}
                >
                  {t.label}
                </button>
              );
            })}
          </div>}
          <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
            {middleTab === "summary" &&
              (pipelineSummary ? (
                <SummaryView
                  data={pipelineSummary}
                  rangeLabel={
                    range[0] === 0 && range[1] === TIME_BINS - 1
                      ? "full recording"
                      : (() => {
                          const span = TIME_BINS;
                          const tFull = meta.duration_ns;
                          const lo = (range[0] / span) * tFull;
                          const hi = ((range[1] + 1) / span) * tFull;
                          const fmt = (n: number) => {
                            const s = n / 1e9;
                            return s >= 1
                              ? `${s.toFixed(2)}s`
                              : `${(s * 1000).toFixed(0)}ms`;
                          };
                          return `${fmt(lo)} – ${fmt(hi)} · ${fmt(hi - lo)} window`;
                        })()
                  }
                />
              ) : (
                <Spinner label="Loading pipeline summary…" />
              ))}
            {middleTab === "flame" && (
              <>
                {flameLoading && <Spinner label="Loading flamegraph…" />}
                {flame && !flameLoading && (
                  <FlameGraph
                    data={flame}
                    selectedFrame={selectedFrame}
                    onSelectFrame={onSelectFrame}
                  />
                )}
              </>
            )}
          </div>
        </div>
      )}

      {selectedCat && selectedFrame && <Splitter onResize={onResizeFlame} />}

      {selectedCat && selectedFrame && (
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              padding: "8px 14px",
              borderBottom: "1px solid oklch(0.93 0.005 250)",
              display: "flex",
              alignItems: "center",
              gap: 12,
              background: "white",
            }}
          >
            <span
              style={{
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: 0.5,
                color: "oklch(0.45 0.01 250)",
              }}
            >
              Inside
            </span>
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: 12.5,
                color: "oklch(0.2 0.01 250)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {selectedFrame}
            </span>
            <div style={{ flex: 1 }} />
            <ModeToggle
              label="Cycles"
              value={cyclesMode}
              onChange={setCyclesMode}
              options={[
                { id: "relative", label: "Relative %" },
                { id: "absolute", label: "Absolute %" },
                { id: "raw", label: "Raw" },
              ]}
            />
            <ModeToggle
              label="Memory"
              value={memoryMode}
              onChange={setMemoryMode}
              options={[
                { id: "relative", label: "Relative %" },
                { id: "weighted", label: "Weighted" },
                { id: "raw", label: "Raw" },
              ]}
            />
            <button
              onClick={() => {
                setSelectedFrame(null);
                setAnnotation(null);
                setAnnotationError(null);
              }}
              style={{
                border: "1px solid oklch(0.9 0.005 250)",
                background: "white",
                borderRadius: 4,
                padding: "3px 9px",
                fontSize: 11,
                color: "oklch(0.45 0.01 250)",
                cursor: "pointer",
              }}
            >
              Close
            </button>
          </div>
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            {annotationLoading && <Spinner label="Loading source + asm…" />}
            {annotationError && (
              <div
                style={{
                  padding: 12,
                  color: "oklch(0.4 0.18 25)",
                  fontSize: 12,
                }}
              >
                {annotationError}
              </div>
            )}
            {annotation && !annotationLoading && (
              <CodePanel
                data={annotation}
                cyclesMode={cyclesMode}
                setCyclesMode={setCyclesMode}
                memoryMode={memoryMode}
                setMemoryMode={setMemoryMode}
                totalProfileCycles={annotation.totals.cycles}
                onAsmSelect={setActiveAsm}
              />
            )}
          </div>
        </div>
      )}

      {/* Per-instruction detail sidebar — overlays the right edge while
          an asm row is selected. */}
      {activeAsm && (
        <InstructionDetail
          symbol={annotation?.symbol ?? null}
          offset={activeAsm.offset ?? null}
          disasm={activeAsm.disasm ?? null}
          addr={activeAsm.addr ?? null}
          onClose={() => setActiveAsm(null)}
        />
      )}
    </div>
  );
}
