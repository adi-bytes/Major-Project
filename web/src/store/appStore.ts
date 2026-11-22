import { create } from "zustand";

export type TabId =
  | "overview"
  | "forward"
  | "reverse"
  | "validation"
  | "model"
  | "design";

interface AppState {
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  // Forward prediction sliders
  wf: number;
  temperature: number;
  setWf: (v: number) => void;
  setTemperature: (v: number) => void;

  // Reverse mode
  reverseInputV: number[];
  reverseInputI: number[];
  setReverseInput: (v: number[], i: number[]) => void;
  revealGroundTruth: boolean;
  setRevealGroundTruth: (v: boolean) => void;
  selectedPreset: string | null;
  setSelectedPreset: (name: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: "overview",
  setActiveTab: (tab) => set({ activeTab: tab }),

  wf: 5.09,
  temperature: 400,
  setWf: (v) => set({ wf: v }),
  setTemperature: (v) => set({ temperature: v }),

  reverseInputV: [],
  reverseInputI: [],
  setReverseInput: (v, i) => set({ reverseInputV: v, reverseInputI: i }),
  revealGroundTruth: false,
  setRevealGroundTruth: (v) => set({ revealGroundTruth: v }),
  selectedPreset: null,
  setSelectedPreset: (name) => set({ selectedPreset: name, revealGroundTruth: false }),
}));
