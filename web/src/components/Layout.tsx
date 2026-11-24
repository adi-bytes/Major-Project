import type { ReactNode } from "react";
import { HonestyBanner } from "./HonestyBanner";
import { useAppStore, type TabId } from "@/store/appStore";
import { useHealth } from "@/api/client";

const TABS: { id: TabId; label: string; experimental?: boolean }[] = [
  { id: "overview", label: "Overview / Method" },
  { id: "forward", label: "Forward Prediction" },
  { id: "reverse", label: "Reverse Mode" },
  { id: "validation", label: "Validation Dashboard" },
  { id: "model", label: "Model & Data Details" },
  { id: "design", label: "Design Workbench" },
];

export function Layout({ children }: { children: ReactNode }) {
  const { activeTab, setActiveTab } = useAppStore();
  const { data: health } = useHealth();

  return (
    <div className="min-h-screen flex flex-col bg-gray-950">
      {/* Permanent honesty banner */}
      <HonestyBanner />

      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-bold text-brand-200 leading-tight">
              Physics-Guided Power Semiconductor Inverse Design Framework
            </h1>
            <p className="text-xs text-gray-400">
              Adaptive Multi-Material Pipeline · Initial Reference Vehicle: Vertical β-Ga₂O₃ SBD
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {health ? (
              <span className="badge badge-green">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                API Connected
              </span>
            ) : (
              <span className="badge badge-red">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 inline-block" />
                API Offline
              </span>
            )}
            <span className="badge badge-blue">TCAD Validated</span>
          </div>
        </div>

        {/* Tab navigation */}
        <nav className="max-w-7xl mx-auto px-4 flex gap-1 overflow-x-auto pb-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={[
                "px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap border-b-2",
                activeTab === tab.id
                  ? "border-brand-500 text-brand-300 bg-gray-800/60"
                  : "border-transparent text-gray-400 hover:text-gray-200 hover:bg-gray-800/30",
                tab.experimental ? "italic" : "",
              ].join(" ")}
            >
              {tab.label}
              {tab.experimental && (
                <span className="ml-1.5 text-[10px] text-yellow-500 font-normal not-italic">
                  ⚗️
                </span>
              )}
            </button>
          ))}
        </nav>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 bg-gray-900/60 text-xs text-gray-600 text-center py-3">
        Data: Dhillon et al., IEEE TED 68(11), 2021 · Sentaurus TCAD ·{" "}
        <span className="text-yellow-700">Not for fabrication use</span>
      </footer>
    </div>
  );
}
