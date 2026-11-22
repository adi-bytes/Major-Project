import { useQuery, useMutation } from "@tanstack/react-query";
import type {
  DeviceMetaResponse,
  HealthResponse,
  PredictRequest,
  PredictResponse,
  ReverseResponse,
  ValidationResponse,
} from "./types";

const BASE = "/api";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail) ?? "API error"
    );
  }
  return res.json() as Promise<T>;
}

// --- Hooks ---

export function useHealth() {
  return useQuery<HealthResponse, Error>({
    queryKey: ["health"],
    queryFn: () => apiFetch<HealthResponse>("/health"),
    retry: 1,
    staleTime: 10_000,
  });
}

export function useMeta() {
  return useQuery<DeviceMetaResponse, Error>({
    queryKey: ["meta"],
    queryFn: () => apiFetch<DeviceMetaResponse>("/meta"),
    staleTime: Infinity,
  });
}

export function useValidation() {
  return useQuery<ValidationResponse, Error>({
    queryKey: ["validation"],
    queryFn: () => apiFetch<ValidationResponse>("/validation"),
    staleTime: Infinity,
  });
}

export function usePrediction(params: PredictRequest | null) {
  return useQuery<PredictResponse, Error>({
    queryKey: ["predict", params],
    queryFn: () =>
      apiFetch<PredictResponse>("/predict", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    enabled: params !== null,
    staleTime: 60_000,
    retry: 1,
  });
}

export function useReverse() {
  return useMutation<ReverseResponse, Error, { v: number[]; i: number[] }>({
    mutationFn: (data) =>
      apiFetch<ReverseResponse>("/reverse", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

import type {
  DesignPresetsResponse,
  DesignRequest,
  DesignResponse,
  ExportReportRequest,
  ExportReportResponse,
  ExportTCADRequest,
  ExportTCADResponse,
} from "./types";

export function useDesignPresets() {
  return useQuery<DesignPresetsResponse, Error>({
    queryKey: ["designPresets"],
    queryFn: () => apiFetch<DesignPresetsResponse>("/design/presets"),
    staleTime: Infinity,
  });
}

export function useDesignOptimization() {
  return useMutation<DesignResponse, Error, DesignRequest>({
    mutationFn: (data) =>
      apiFetch<DesignResponse>("/design", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

export function useExportTCAD() {
  return useMutation<ExportTCADResponse, Error, ExportTCADRequest>({
    mutationFn: (data) =>
      apiFetch<ExportTCADResponse>("/export/tcad", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

export function useExportReport() {
  return useMutation<ExportReportResponse, Error, ExportReportRequest>({
    mutationFn: (data) =>
      apiFetch<ExportReportResponse>("/export/report", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}
