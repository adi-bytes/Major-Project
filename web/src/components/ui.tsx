interface LoadingSpinnerProps {
  text?: string;
}

export function LoadingSpinner({ text = "Loading…" }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-gray-400">
      <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm">{text}</span>
    </div>
  );
}

interface ErrorBoxProps {
  message: string;
  retry?: () => void;
}

export function ErrorBox({ message, retry }: ErrorBoxProps) {
  return (
    <div className="rounded-xl border border-red-800 bg-red-950/40 p-5 text-sm">
      <p className="text-red-300 font-semibold mb-1">Error</p>
      <p className="text-red-200">{message}</p>
      {retry && (
        <button
          onClick={retry}
          className="mt-3 px-4 py-1.5 rounded-lg bg-red-900/60 hover:bg-red-900 text-red-200 text-xs transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}

interface WarnBoxProps {
  items: string[];
}

export function WarnBox({ items }: WarnBoxProps) {
  if (!items.length) return null;
  return (
    <div className="rounded-xl border border-yellow-700 bg-yellow-950/40 p-4 text-sm">
      <p className="text-yellow-300 font-semibold mb-1">⚠️ Warnings</p>
      <ul className="list-disc list-inside space-y-0.5 text-yellow-200">
        {items.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}
