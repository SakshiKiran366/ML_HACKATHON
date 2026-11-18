import type { HealthStatus, ReconstructionResult, TripData, TripSummary } from '../types'

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export const API_CONTRACT = {
  health: '/health',
  trips: '/trips',
  trip: (id: string) => `/trips/${encodeURIComponent(id)}`,
  reconstruct: '/reconstruct',
} as const

export function normalizeBaseUrl(value: string) {
  return value.trim().replace(/\/$/, '')
}

export function getConfiguredBaseUrl() {
  const saved = typeof window !== 'undefined' ? window.localStorage.getItem('pathsnap_api_base_url') : null
  return normalizeBaseUrl(saved || configuredBaseUrl)
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 12000)

  try {
    const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    })
    const body = await response.text()
    let payload: unknown = null
    try {
      payload = body ? JSON.parse(body) : null
    } catch {
      payload = body
    }
    if (!response.ok) {
      const detail = typeof payload === 'object' && payload !== null && 'detail' in payload ? String(payload.detail) : `HTTP ${response.status}`
      throw new Error(detail)
    }
    return payload as T
  } finally {
    window.clearTimeout(timer)
  }
}

function toTripSummary(item: unknown): TripSummary | null {
  if (typeof item === 'string' || typeof item === 'number') return { id: String(item) }
  if (!item || typeof item !== 'object') return null
  const value = item as Record<string, unknown>
  const id = value.id ?? value.trip_id ?? value.tripId
  return id === undefined ? null : { id: String(id), label: typeof value.label === 'string' ? value.label : undefined, observationCount: typeof value.observation_count === 'number' ? value.observation_count : undefined }
}

function normalizeTrip(payload: unknown, fallbackId?: string): TripData {
  const value = (payload && typeof payload === 'object' ? payload : {}) as Record<string, unknown>
  const id = String(value.id ?? value.trip_id ?? fallbackId ?? 'unknown')
  const observations = (value.observations ?? value.gps_trajectory ?? value.gps ?? []) as unknown
  const predictedRoute = (value.predicted_route ?? value.route ?? value.geometry ?? []) as unknown
  const roadNetwork = (value.road_network ?? value.road_segments ?? []) as unknown
  return {
    id,
    observations: normalizeGeometry(observations),
    predictedRoute: normalizeGeometry(predictedRoute),
    roadNetwork: normalizeMultiGeometry(roadNetwork),
    start: normalizeCoordinate(value.start),
    end: normalizeCoordinate(value.end),
    metrics: normalizeMetrics(value.metrics ?? value.evaluation),
  }
}

function normalizeCoordinate(value: unknown): [number, number] | undefined {
  return Array.isArray(value) && value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1])) ? [Number(value[0]), Number(value[1])] : undefined
}

function normalizeGeometry(value: unknown): [number, number][] {
  if (value && typeof value === 'object' && 'coordinates' in value) return normalizeGeometry((value as { coordinates?: unknown }).coordinates)
  if (!Array.isArray(value)) return []
  return value.map(normalizeCoordinate).filter((item): item is [number, number] => Boolean(item))
}

function normalizeMultiGeometry(value: unknown): [number, number][][] {
  if (!Array.isArray(value)) return []
  return value.map((segment) => normalizeGeometry(segment)).filter((segment) => segment.length > 1)
}

function normalizeMetrics(value: unknown): Record<string, number | string | null> | undefined {
  if (!value || typeof value !== 'object') return undefined
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).filter(([, item]) => ['number', 'string'].includes(typeof item) || item === null)) as Record<string, number | string | null>
}

export async function checkHealth(baseUrl: string): Promise<HealthStatus> {
  const started = performance.now()
  try {
    await request<unknown>(baseUrl, API_CONTRACT.health)
    return { ok: true, message: 'Backend responding', latencyMs: Math.round(performance.now() - started) }
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : 'Backend unavailable' }
  }
}

export async function fetchTrips(baseUrl: string): Promise<TripSummary[]> {
  const payload = await request<unknown>(baseUrl, API_CONTRACT.trips)
  const list = Array.isArray(payload) ? payload : (payload as Record<string, unknown>)?.trips
  return Array.isArray(list) ? list.map(toTripSummary).filter((item): item is TripSummary => Boolean(item)) : []
}

export async function fetchTrip(baseUrl: string, id: string): Promise<TripData> {
  return normalizeTrip(await request<unknown>(baseUrl, API_CONTRACT.trip(id)), id)
}

export async function reconstructTrip(baseUrl: string, id: string): Promise<ReconstructionResult> {
  const payload = await request<unknown>(baseUrl, API_CONTRACT.reconstruct, { method: 'POST', body: JSON.stringify({ trip_id: id }) })
  const result = normalizeTrip(payload, id)
  const record = (payload && typeof payload === 'object' ? payload : {}) as Record<string, unknown>
  return { ...result, processingTimeMs: typeof record.processing_time_ms === 'number' ? record.processing_time_ms : undefined }
}
