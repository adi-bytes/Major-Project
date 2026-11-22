export function HonestyBanner() {
  return (
    <div
      className="w-full bg-yellow-950 border-b border-yellow-700 px-4 py-2 text-center text-xs text-yellow-300 font-medium tracking-wide z-50"
      role="alert"
      aria-live="polite"
    >
      <span className="font-bold">[HONESTY] </span>
      Research prototype trained on TCAD simulation data. Not validated for
      fabrication or experimental device design.
    </div>
  );
}
