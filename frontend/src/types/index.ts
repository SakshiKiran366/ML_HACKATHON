export type AppMode = 'demo' | 'live'

export type Coordinate = [number, number] // [longitude, latitude]

export interface TripSummary {
  id: string
  label?: string
  observationCount?: number
}

export interface TripData {
  id: string
  observations: Coordinate[]
  roadNetwork?: Coordinate[][]
  predictedRoute?: Coordinate[]
  start?: Coordinate
  end?: Coordinate
  metrics?: Record<string, number | string | null>
}

export interface ReconstructionResult extends TripData {
  processingTimeMs?: number
}

export interface HealthStatus {
  ok: boolean
  message: string
  latencyMs?: number
}
