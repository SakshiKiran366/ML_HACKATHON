import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from 'react-leaflet'
import { Activity, AlertCircle, ArrowDownToLine, Check, ChevronDown, CircleHelp, Crosshair, Database, ExternalLink, Gauge, Layers3, LoaderCircle, Map as MapIcon, Maximize2, Network, PanelLeftClose, PanelLeftOpen, Play, Radio, RefreshCw, Route, Save, Settings, ShieldCheck, SlidersHorizontal, Sparkles, Target, X, Zap } from 'lucide-react'
import { checkHealth, fetchTrip, fetchTrips, getConfiguredBaseUrl, normalizeBaseUrl, reconstructTrip } from './services/api'
import type { AppMode, Coordinate, HealthStatus, ReconstructionResult, TripData, TripSummary } from './types'

const DEMO_TRIPS: TripData[] = [
  {
    id: 'DEMO-01',
    observations: [[77.594, 12.971], [77.598, 12.973], [77.601, 12.975], [77.604, 12.978], [77.607, 12.981], [77.611, 12.984], [77.615, 12.985], [77.619, 12.988], [77.622, 12.992]],
    predictedRoute: [[77.594, 12.971], [77.599, 12.973], [77.604, 12.977], [77.609, 12.981], [77.614, 12.984], [77.619, 12.988], [77.622, 12.992]],
    roadNetwork: [[[77.59, 12.968], [77.604, 12.977], [77.616, 12.976], [77.624, 12.991]], [[77.599, 12.978], [77.609, 12.981], [77.618, 12.997]], [[77.602, 12.964], [77.612, 12.975], [77.629, 12.982]]],
  },
  {
    id: 'DEMO-02',
    observations: [[77.578, 12.955], [77.581, 12.958], [77.586, 12.96], [77.59, 12.964], [77.593, 12.967], [77.598, 12.969], [77.603, 12.971], [77.606, 12.975]],
    predictedRoute: [[77.578, 12.955], [77.584, 12.959], [77.59, 12.963], [77.597, 12.968], [77.606, 12.975]],
    roadNetwork: [[[77.574, 12.951], [77.59, 12.963], [77.606, 12.975]], [[77.58, 12.968], [77.59, 12.963], [77.611, 12.964]], [[77.59, 12.963], [77.6, 12.955]]],
  },
  {
    id: 'DEMO-03',
    observations: [[77.626, 12.984], [77.624, 12.988], [77.62, 12.991], [77.616, 12.994], [77.611, 12.997], [77.607, 13.001], [77.603, 13.005]],
    predictedRoute: [[77.626, 12.984], [77.621, 12.99], [77.616, 12.995], [77.61, 13.0], [77.603, 13.005]],
    roadNetwork: [[[77.631, 12.98], [77.621, 12.99], [77.61, 13.0], [77.6, 13.009]], [[77.617, 12.978], [77.621, 12.99], [77.635, 13.001]], [[77.621, 12.99], [77.611, 12.997]]],
  },
]

const DEMO_SUMMARIES: TripSummary[] = DEMO_TRIPS.map((trip) => ({ id: trip.id, label: 'Illustrative sample', observationCount: trip.observations.length }))

const metricLabels: Record<string, { label: string; unit?: string; description: string }> = {
  accuracy: { label: 'Route accuracy', unit: '%', description: 'Backend-reported match accuracy.' },
  precision: { label: 'Precision', unit: '%', description: 'Correct predicted route segments.' },
  recall: { label: 'Recall', unit: '%', description: 'Reference route segments recovered by the model.' },
  f1: { label: 'F1 score', description: 'Harmonic mean of precision and recall.' },
  route_similarity: { label: 'Route similarity', unit: '%', description: 'Backend-reported similarity score.' },
  distance_error: { label: 'Distance error', unit: 'm', description: 'Backend-reported distance-based error.' },
  processing_time_ms: { label: 'Processing time', unit: 'ms', description: 'Time reported by the inference service.' },
}

function demoTrip(id: string) {
  return DEMO_TRIPS.find((trip) => trip.id === id) ?? DEMO_TRIPS[0]
}

function downsample(points: Coordinate[], max = 1200) {
  if (points.length <= max) return points
  const stride = Math.ceil(points.length / max)
  return points.filter((_, index) => index % stride === 0 || index === points.length - 1)
}

function toLatLng(points: Coordinate[]) {
  return downsample(points).map(([longitude, latitude]) => [latitude, longitude] as [number, number])
}

function boundsFor(trip?: TripData | null): [number, number][] {
  if (!trip) return []
  const points = [...trip.observations, ...(trip.predictedRoute ?? [])]
  return points.map(([longitude, latitude]) => [latitude, longitude])
}

function MapViewport({ points, resetToken }: { points: [number, number][]; resetToken: number }) {
  const map = useMap()
  useEffect(() => {
    if (points.length > 1) map.fitBounds(points, { padding: [28, 28], maxZoom: 16 })
  }, [map, points])
  useEffect(() => {
    if (resetToken > 0 && points.length > 1) map.fitBounds(points, { padding: [28, 28], maxZoom: 16 })
  }, [map, points, resetToken])
  return null
}

function MapTools({ onFit, onFullscreen, onReset }: { onFit: () => void; onFullscreen: () => void; onReset: () => void }) {
  return <div className="map-tools" aria-label="Map controls">
    <button onClick={onFit} title="Fit selected trip"><Crosshair size={15} /><span>Fit route</span></button>
    <button onClick={onReset} title="Reset map view"><RefreshCw size={15} /><span>Reset</span></button>
    <button onClick={onFullscreen} title="Enter fullscreen"><Maximize2 size={15} /><span>Fullscreen</span></button>
  </div>
}

function RouteMap({ trip, layers, setLayers }: { trip: TripData | null; layers: { gps: boolean; road: boolean; prediction: boolean; markers: boolean }; setLayers: (next: { gps: boolean; road: boolean; prediction: boolean; markers: boolean }) => void }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [resetToken, setResetToken] = useState(0)
  const points = useMemo(() => boundsFor(trip), [trip])
  const center = points[Math.floor(points.length / 2)] ?? [12.971, 77.604]
  const start = trip?.start ? [trip.start[1], trip.start[0]] as [number, number] : toLatLng(trip?.observations.slice(0, 1) ?? [])[0]
  const end = trip?.end ? [trip.end[1], trip.end[0]] as [number, number] : toLatLng(trip?.observations.slice(-1) ?? [])[0]
  const fit = useCallback(() => {
    setResetToken((token) => token + 1)
  }, [])

  return <div ref={containerRef} className="map-frame">
    <MapContainer center={center as [number, number]} zoom={14} zoomControl={false} scrollWheelZoom className="pathsnap-map">
      <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <MapViewport points={points} resetToken={resetToken} />
      {layers.road && trip?.roadNetwork?.map((segment, index) => <Polyline key={`road-${index}`} positions={toLatLng(segment)} pathOptions={{ color: '#718198', weight: 2, opacity: 0.45, dashArray: '3 6' }} />)}
      {layers.gps && trip && <Polyline positions={toLatLng(trip.observations)} pathOptions={{ color: '#f2b84b', weight: 3, opacity: 0.9 }} />}
      {layers.prediction && trip?.predictedRoute && trip.predictedRoute.length > 1 && <Polyline positions={toLatLng(trip.predictedRoute)} pathOptions={{ color: '#4bd2a0', weight: 5, opacity: 1 }} />}
      {layers.markers && start && <CircleMarker center={start} radius={7} pathOptions={{ color: '#081019', weight: 3, fillColor: '#4bd2a0', fillOpacity: 1 }}><Popup>Start observation</Popup></CircleMarker>}
      {layers.markers && end && <CircleMarker center={end} radius={7} pathOptions={{ color: '#081019', weight: 3, fillColor: '#ee6d7a', fillOpacity: 1 }}><Popup>End observation</Popup></CircleMarker>}
    </MapContainer>
    <MapTools onFit={fit} onReset={fit} onFullscreen={() => containerRef.current?.requestFullscreen?.()} />
    <div className="map-layers">
      <div className="map-layers-heading"><Layers3 size={15} /> Map layers</div>
      {([['gps', 'GPS trajectory', 'amber'], ['road', 'Road context', 'slate'], ['prediction', 'Predicted route', 'mint'], ['markers', 'Start / end', 'rose']] as const).map(([key, label, tone]) => <label className="layer-toggle" key={key}><input type="checkbox" checked={layers[key]} onChange={() => setLayers({ ...layers, [key]: !layers[key] })} /><span className={`layer-swatch ${tone}`} />{label}</label>)}
    </div>
    <div className="map-attribution">Visualization geometry is limited to the selected trip.</div>
  </div>
}

function StatusPill({ mode, health }: { mode: AppMode; health: HealthStatus }) {
  if (mode === 'demo') return <span className="status-pill demo"><span className="status-dot" />Demo mode</span>
  return <span className={`status-pill ${health.ok ? 'live' : 'offline'}`}><span className="status-dot" />{health.ok ? 'Backend connected' : 'Backend offline'}</span>
}

function ApiSettings({ baseUrl, health, onClose, onSave }: { baseUrl: string; health: HealthStatus; onClose: () => void; onSave: (url: string) => Promise<void> }) {
  const [draft, setDraft] = useState(baseUrl)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const submit = async () => {
    try {
      const parsed = new URL(draft)
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Use an http:// or https:// URL.')
      setError(''); setSaving(true); await onSave(normalizeBaseUrl(parsed.toString()))
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Could not save this URL.')
    } finally { setSaving(false) }
  }
  return <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="modal-heading"><div><span className="eyebrow">Connection</span><h2 id="settings-title">API settings</h2></div><button className="icon-button" onClick={onClose} aria-label="Close settings"><X size={18} /></button></div>
      <p className="modal-copy">Point PathSnap at the FastAPI service running on the backend laptop. The URL is saved locally in this browser.</p>
      <label className="field-label" htmlFor="api-url">Backend base URL</label>
      <input id="api-url" className="text-input" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="http://192.168.1.24:8000" />
      {error && <div className="inline-error"><AlertCircle size={15} />{error}</div>}
      <div className={`connection-box ${health.ok ? 'ok' : 'bad'}`}><div className="connection-icon">{health.ok ? <Check size={17} /> : <AlertCircle size={17} />}</div><div><strong>{health.ok ? 'Connection healthy' : 'Connection unavailable'}</strong><span>{health.message}{health.latencyMs ? ` · ${health.latencyMs} ms` : ''}</span></div></div>
      <div className="modal-actions"><button className="button secondary" onClick={onClose}>Cancel</button><button className="button primary" onClick={submit} disabled={saving}>{saving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}Save & test</button></div>
      <p className="settings-note"><ShieldCheck size={14} />No secrets or protected evaluation data are stored in the frontend.</p>
    </section>
  </div>
}

function MetricsPanel({ result, mode }: { result: ReconstructionResult | null; mode: AppMode }) {
  const metrics = result?.metrics ? Object.entries(result.metrics).filter(([, value]) => value !== null && value !== '') : []
  return <section className="panel metrics-panel">
    <div className="panel-heading"><div><span className="eyebrow">Evaluation</span><h2>Model performance</h2></div><Gauge size={18} className="panel-icon" /></div>
    {mode === 'demo' && <div className="metrics-empty"><Sparkles size={20} /><strong>Live evaluation appears here</strong><p>This preview uses illustrative geometry only. No model metrics are being presented in demo mode.</p></div>}
    {mode === 'live' && !result && <div className="metrics-empty"><Target size={20} /><strong>Run a reconstruction</strong><p>Backend-reported metrics will appear after inference completes.</p></div>}
    {mode === 'live' && result && metrics.length === 0 && <div className="metrics-empty"><AlertCircle size={20} /><strong>No metrics returned</strong><p>The backend returned a route without evaluation fields for this trip.</p></div>}
    {metrics.length > 0 && <div className="metric-grid">{metrics.map(([key, value]) => { const definition = metricLabels[key] ?? { label: key.replaceAll('_', ' '), description: 'Returned by the backend.' }; return <div className="metric" key={key}><span className="metric-label">{definition.label}<span title={definition.description}><CircleHelp size={12} /></span></span><strong>{String(value)}{definition.unit ? <small>{definition.unit}</small> : null}</strong></div> })}</div>}
    {result?.processingTimeMs && <div className="processing-row"><Zap size={14} />Inference completed in <strong>{result.processingTimeMs} ms</strong></div>}
  </section>
}

export default function App() {
  const [mode, setMode] = useState<AppMode>('demo')
  const [baseUrl, setBaseUrl] = useState(getConfiguredBaseUrl())
  const [health, setHealth] = useState<HealthStatus>({ ok: false, message: 'Not tested yet' })
  const [trips, setTrips] = useState<TripSummary[]>(DEMO_SUMMARIES)
  const [selectedId, setSelectedId] = useState(DEMO_SUMMARIES[0].id)
  const [trip, setTrip] = useState<TripData | null>(DEMO_TRIPS[0])
  const [result, setResult] = useState<ReconstructionResult | null>(null)
  const [loadingTrips, setLoadingTrips] = useState(false)
  const [loadingTrip, setLoadingTrip] = useState(false)
  const [reconstructing, setReconstructing] = useState(false)
  const [error, setError] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [layers, setLayers] = useState({ gps: true, road: true, prediction: true, markers: true })

  const loadLiveTrips = useCallback(async (url = baseUrl) => {
    setLoadingTrips(true); setError('')
    const nextHealth = await checkHealth(url); setHealth(nextHealth)
    if (!nextHealth.ok) { setTrips([]); setTrip(null); setLoadingTrips(false); return }
    try {
      const availableTrips = await fetchTrips(url)
      setTrips(availableTrips)
      setSelectedId((current) => availableTrips.some((item) => item.id === current) ? current : (availableTrips[0]?.id ?? ''))
      if (!availableTrips.length) setTrip(null)
    } catch (loadError) {
      setTrips([]); setTrip(null); setError(loadError instanceof Error ? loadError.message : 'Could not load trips.')
    } finally { setLoadingTrips(false) }
  }, [baseUrl])

  useEffect(() => { if (mode === 'live') void loadLiveTrips() }, [mode, loadLiveTrips])

  useEffect(() => {
    if (!selectedId) return
    if (mode === 'demo') { setTrip(demoTrip(selectedId)); setResult(null); return }
    setLoadingTrip(true); setError(''); setResult(null)
    void fetchTrip(baseUrl, selectedId).then(setTrip).catch((loadError) => { setTrip(null); setError(loadError instanceof Error ? loadError.message : 'Could not load this trip.') }).finally(() => setLoadingTrip(false))
  }, [baseUrl, mode, selectedId])

  const saveApiUrl = async (url: string) => {
    localStorage.setItem('pathsnap_api_base_url', url)
    setBaseUrl(url)
    await loadLiveTrips(url)
  }

  const switchMode = (nextMode: AppMode) => {
    setMode(nextMode); setError(''); setResult(null)
    if (nextMode === 'demo') { setTrips(DEMO_SUMMARIES); setSelectedId(DEMO_SUMMARIES[0].id); setTrip(DEMO_TRIPS[0]); setHealth({ ok: false, message: 'Demo mode is active' }) }
    if (nextMode === 'live') { setSelectedId(''); setTrip(null) }
  }

  const reconstruct = async () => {
    if (!selectedId || reconstructing) return
    if (mode === 'demo') { setResult(null); setError('Demo mode is illustrative only. Switch to Live mode to run backend inference.'); return }
    setReconstructing(true); setError('');
    try { const response = await reconstructTrip(baseUrl, selectedId); setResult(response); setTrip(response) } catch (reconstructError) { setError(reconstructError instanceof Error ? reconstructError.message : 'Reconstruction request failed.') } finally { setReconstructing(false) }
  }

  const selectedSummary = trips.find((item) => item.id === selectedId)
  const mapStatus = loadingTrip ? 'Loading selected trip…' : trip ? `${trip.observations.length.toLocaleString()} GPS observations loaded` : 'No trip geometry loaded'

  return <div className="app-shell">
    <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
      <div className="brand-row"><div className="brand-mark"><Route size={19} /></div>{sidebarOpen && <div><div className="brand-name">Path<span>Snap</span></div><div className="brand-subtitle">Route intelligence</div></div>}<button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Toggle sidebar">{sidebarOpen ? <PanelLeftClose size={17} /> : <PanelLeftOpen size={17} />}</button></div>
      {sidebarOpen && <>
        <div className="sidebar-section-label">Workspace</div>
        <nav className="nav-list"><button className="nav-item active"><MapIcon size={17} /><span>Reconstruction lab</span><span className="nav-dot" /></button><button className="nav-item"><Database size={17} /><span>Trip catalog</span></button><button className="nav-item"><Network size={17} /><span>Road context</span></button></nav>
        <div className="sidebar-section-label mode-label">Runtime mode</div>
        <div className="mode-switch"><button className={mode === 'demo' ? 'selected' : ''} onClick={() => switchMode('demo')}><Sparkles size={14} />Demo</button><button className={mode === 'live' ? 'selected' : ''} onClick={() => switchMode('live')}><Radio size={14} />Live</button></div>
        <div className="sidebar-note"><div className="note-icon"><ShieldCheck size={15} /></div><div><strong>Privacy by design</strong><p>Ground truth stays server-side. The browser receives permitted geometry and metrics only.</p></div></div>
        <div className="sidebar-footer"><div className="version">PATHSNAP <span>v0.1</span></div><button className="nav-item footer-settings" onClick={() => setSettingsOpen(true)}><Settings size={16} /><span>API settings</span></button></div>
      </>}
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="topbar-title"><span className="breadcrumb">Workspace / <strong>Reconstruction lab</strong></span><span className="live-clock"><span className={`status-dot ${mode === 'live' && health.ok ? 'good' : ''}`} />{mode === 'demo' ? 'Preview environment' : health.ok ? 'Inference service online' : 'Waiting for backend'}</span></div><div className="topbar-actions"><StatusPill mode={mode} health={health} /><button className="icon-button top-settings" onClick={() => setSettingsOpen(true)} aria-label="Open API settings"><SlidersHorizontal size={18} /></button><div className="avatar">PS</div></div></header>

      <div className="content-wrap">
        <section className="page-intro"><div><div className="eyebrow"><span className="eyebrow-line" /> DISTRIBUTED MAP-MATCHING</div><h1>Reconstruct the route.<br /><em>See the signal.</em></h1><p>Inspect noisy GPS trajectories, review road context, and request a server-side prediction from one focused workspace.</p></div><div className="intro-kpis"><div><span>Mode</span><strong>{mode === 'demo' ? 'Illustrative' : health.ok ? 'Connected' : 'Offline'}</strong></div><div><span>Trip set</span><strong>{loadingTrips ? '…' : trips.length || '—'} <small>available</small></strong></div></div></section>
        {mode === 'demo' && <div className="demo-banner"><Sparkles size={15} /><span><strong>Demo mode</strong> — geometry below is illustrative and is not a model prediction or evaluation result.</span><button onClick={() => switchMode('live')}>Connect a backend <ExternalLink size={13} /></button></div>}
        {error && <div className="error-banner"><AlertCircle size={17} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}

        <div className="workflow-grid">
          <section className="panel control-panel"><div className="panel-heading"><div><span className="eyebrow">01 / INPUT</span><h2>Select a trip</h2></div><span className="step-number">01</span></div><label className="field-label" htmlFor="trip-select">Evaluation trip</label><div className="select-wrap"><select id="trip-select" value={selectedId} onChange={(event) => setSelectedId(event.target.value)} disabled={loadingTrips || !trips.length}>{!trips.length && <option value="">No trips available</option>}{trips.map((item) => <option key={item.id} value={item.id}>{item.id}{item.label ? ` — ${item.label}` : ''}</option>)}</select><ChevronDown size={17} /></div>{selectedSummary && <div className="trip-meta"><div className="meta-row"><span>Trip status</span><span className="meta-value ready"><span className="status-dot" />Ready to inspect</span></div><div className="meta-row"><span>Observations</span><strong>{selectedSummary.observationCount ?? trip?.observations.length ?? '—'}</strong></div><div className="meta-row"><span>Data source</span><strong>{mode === 'demo' ? 'Illustrative sample' : 'FastAPI backend'}</strong></div></div>}{!trips.length && mode === 'live' && <div className="empty-trips"><Database size={18} /><strong>No available trips</strong><p>Check the backend connection and confirm the trips endpoint is populated.</p></div>}<div className="control-divider" /><div className="panel-heading compact"><div><span className="eyebrow">02 / INFERENCE</span><h2>Reconstruction</h2></div><Zap size={17} className="panel-icon" /></div><p className="control-copy">The backend will perform map matching and return the permitted predicted route geometry and evaluation fields.</p><button className="button primary reconstruct-button" onClick={reconstruct} disabled={!selectedId || reconstructing || loadingTrip || mode === 'demo'}>{reconstructing ? <><LoaderCircle size={17} className="spin" />Processing route…</> : <><Play size={16} fill="currentColor" />Reconstruct route</>}</button>{mode === 'demo' && <span className="disabled-hint">Switch to Live mode to call the inference API.</span>}</section>

          <section className="panel map-panel"><div className="map-panel-heading"><div><span className="eyebrow">03 / VISUALIZE</span><h2>Route workspace</h2><span className="map-status"><span className="status-dot good" />{mapStatus}</span></div><div className="map-legend"><span><i className="legend-line amber" />GPS</span><span><i className="legend-line mint" />Prediction</span><span><i className="legend-line slate" />Road context</span></div></div><RouteMap trip={trip} layers={layers} setLayers={setLayers} /></section>
        </div>
        <div className="bottom-grid"><MetricsPanel result={result} mode={mode} /><section className="panel response-panel"><div className="panel-heading"><div><span className="eyebrow">Response</span><h2>Inference status</h2></div><Activity size={18} className="panel-icon" /></div><div className="response-status"><div className={`response-symbol ${result ? 'success' : reconstructing ? 'working' : ''}`}>{result ? <Check size={18} /> : reconstructing ? <LoaderCircle size={18} className="spin" /> : <ArrowDownToLine size={18} />}</div><div><strong>{result ? 'Prediction received' : reconstructing ? 'Processing request' : 'Awaiting reconstruction'}</strong><p>{result ? 'The latest permitted response is rendered on the map.' : mode === 'demo' ? 'Use the illustrative workspace to inspect the interaction model.' : 'Select a trip and start a server-side reconstruction.'}</p></div></div><div className="response-foot"><span>Protected evaluation data <ShieldCheck size={13} /></span><span>{result ? 'Latest response' : 'No active request'}</span></div></section></div>
        <footer className="page-footer"><span>PathSnap / distributed map-matching</span><span>Frontend prepared for a configurable FastAPI backend</span></footer>
      </div>
    </main>
    {settingsOpen && <ApiSettings baseUrl={baseUrl} health={health} onClose={() => setSettingsOpen(false)} onSave={saveApiUrl} />}
  </div>
}
