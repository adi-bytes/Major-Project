// API types matching FastAPI schemas

export interface VariableBound {
  name: string;
  unit: string;
  min_val: number;
  max_val: number;
  is_sweepable: boolean;
}

export interface DeviceMetaResponse {
  device_name: string;
  material: string;
  description: string;
  evidence_level: string;
  variables: VariableBound[];
  fixed_parameters: Record<string, number>;
  voltage_grid: {
    v_min: number;
    v_max: number;
    num_points: number;
    step_v: number;
  };
}

export interface HealthResponse {
  status: string;
  device: string;
  evidence_level: string;
  artifacts_loaded: Record<string, boolean>;
}

export interface ExtractedMetrics {
  v_o: number;
  r_on_sp: number;
  ss_mV_dec: number;
  i_on_4V: number;
  i_leak_0V: number;
}

export interface PredictRequest {
  wf: number;
  temperature: number;
  v_custom?: number[];
}

export interface PredictResponse {
  V: number[];
  I_mean: number[];
  I_std: number[];
  I_lower: number[];
  I_upper: number[];
  I_phys: number[];
  metrics: ExtractedMetrics;
  flags: string[];
  evidence_level: string;
}

export interface ParameterRecovery {
  WF: number;
  T: number;
  WF_uncertainty: number;
  T_uncertainty: number;
  WF_interval: [number, number];
  T_interval: [number, number];
}

export interface NearestTCAD {
  index: number;
  WF: number;
  T: number;
  distance: number;
}

export interface ReverseResponse {
  recovered: ParameterRecovery;
  fit_residual: number;
  reconstructed: PredictResponse;
  nearest_tcad: NearestTCAD;
  warnings: string[];
  status: string;
  evidence_level: string;
}

export interface ValidationResponse {
  metrics: Record<string, unknown>;
  evidence_level: string;
}


// --- Tier B Design Mode Types ---

export interface DesignCandidate {
  archetype: string;
  verdict: "met" | "met_within_tolerance" | "compromise" | "infeasible";
  verdict_reason: string;
  parameters: {
    WF: number;
    temperature: number;
    t_D_um: number;
    N_net_cm3: number;
    d_um: number;
    anode_area_cm2?: number;
  };
  metrics: {
    v_br: number;
    r_on_sp: number;
    v_o: number;
    ss_mV_dec: number;
    i_on_4V: number;
    i_leak_0V?: number;
    regime?: string;
    w_dep_um?: number;
    bfom_mw_cm2?: number;
  };
  evidence_tags: Record<string, string>;
  iv_curve: {
    V: number[];
    I: number[];
  };
  checks?: Record<string, {
    target: number;
    actual: number;
    unit: string;
    passed: boolean;
    rel_error: number;
  }>;
}

export interface DesignRequest {
  target_v_br?: number | null;
  target_r_on_sp?: number | null;
  target_v_o?: number | null;
  target_ss?: number | null;
  target_i_on_4v?: number | null;
  tolerance?: number;
  method?: "nsga2" | "gradient";
  fixed_temperature?: number | null;
  weights?: Record<string, number>;
}

export interface ParetoPoint {
  v_br: number;
  r_on_sp: number;
  v_o: number;
  i_on_4V: number;
  t_D_um: number;
  N_net_cm3: number;
  archetype?: string;
}

export interface DesignResponse {
  candidates: DesignCandidate[];
  pareto_front: ParetoPoint[];
  requirements: Record<string, unknown>;
  method_used: string;
  evidence_level: string;
  honesty_notice: string;
}

export interface DesignPreset {
  id: string;
  name: string;
  description: string;
  requirements: DesignRequest;
}

export interface DesignPresetsResponse {
  presets: DesignPreset[];
  honesty_notice: string;
}

export interface ExportTCADRequest {
  parameters: Record<string, number>;
  device_name?: string;
}

export interface ExportTCADResponse {
  sdevice_cmd: string;
  sde_cmd: string;
  readme_md: string;
  device_params: Record<string, number>;
  honesty_notice: string;
}

export interface ExportReportRequest {
  candidates: DesignCandidate[];
  requirements: Record<string, unknown>;
}

export interface ExportReportResponse {
  markdown: string;
  title: string;
  honesty_notice: string;
}
