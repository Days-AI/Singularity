interface QueryBarProps {
  query: string;
  onQueryChange: (value: string) => void;
  questions: string[];
  onQuestionsChange: (value: string[]) => void;
  webSourcesEnabled: boolean;
  onToggleWebSources: (value: boolean) => void;
  disabled: boolean;
  onRun: () => void;
}

/**
 * Simulation query input strip. Captures the primary question that drives the
 * run plus an extensible list of optional focus questions that steer report
 * synthesis. Inputs lock while a simulation is connecting or streaming.
 */
export function QueryBar({
  query,
  onQueryChange,
  questions,
  onQuestionsChange,
  webSourcesEnabled,
  onToggleWebSources,
  disabled,
  onRun,
}: QueryBarProps) {
  const updateQuestion = (index: number, value: string) => {
    const next = [...questions];
    next[index] = value;
    onQuestionsChange(next);
  };

  const addQuestion = () => onQuestionsChange([...questions, ""]);

  const removeQuestion = (index: number) =>
    onQuestionsChange(questions.filter((_, i) => i !== index));

  const submit = () => {
    if (disabled || !query.trim()) return;
    onRun();
  };

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-[color:var(--hairline)] bg-panel px-4 py-2.5">
      <div className="flex items-center gap-2">
        <label className="font-mono text-2xs uppercase tracking-widest text-muted">
          Query
        </label>
        <input
          type="text"
          value={query}
          disabled={disabled}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="Analyze market sentiment and behavioral drivers for the next quarter"
          className="min-w-0 flex-1 rounded-sm border border-[color:var(--hairline)] bg-bg px-2.5 py-1 font-mono text-xs text-data outline-none transition-colors focus:border-teal/60 disabled:opacity-50"
        />
        <button
          type="button"
          role="switch"
          aria-checked={webSourcesEnabled}
          onClick={() => onToggleWebSources(!webSourcesEnabled)}
          title="Master switch for all live web sources (Serper, DuckDuckGo, Wikipedia, yFinance). Applies to the next run."
          className={`flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-2xs uppercase tracking-wider transition-colors ${
            webSourcesEnabled
              ? "border-teal/60 bg-teal/10 text-teal"
              : "border-[color:var(--hairline)] text-muted hover:text-data"
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              webSourcesEnabled ? "bg-teal" : "bg-muted"
            }`}
          />
          Web Sources {webSourcesEnabled ? "On" : "Off"}
        </button>
        <button
          onClick={addQuestion}
          disabled={disabled}
          className="rounded-sm border border-[color:var(--hairline)] px-2 py-1 font-mono text-2xs uppercase tracking-wider text-muted transition-colors hover:border-teal/50 hover:text-teal disabled:opacity-50"
        >
          + Question
        </button>
        <button
          onClick={submit}
          disabled={disabled || !query.trim()}
          className="rounded-sm border border-teal/50 bg-teal/10 px-3 py-1 font-mono text-xs font-semibold uppercase tracking-wider text-teal shadow-glow transition-colors hover:bg-teal/20 disabled:opacity-40"
        >
          Run Sim
        </button>
      </div>

      {questions.length > 0 && (
        <div className="flex flex-col gap-1.5 pl-[3.25rem]">
          {questions.map((q, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="font-mono text-2xs text-muted/60">Q{i + 1}</span>
              <input
                type="text"
                value={q}
                disabled={disabled}
                onChange={(e) => updateQuestion(i, e.target.value)}
                placeholder="Optional focus question for the report"
                className="min-w-0 flex-1 rounded-sm border border-[color:var(--hairline)] bg-bg px-2.5 py-1 font-mono text-xs text-data outline-none transition-colors focus:border-teal/60 disabled:opacity-50"
              />
              <button
                onClick={() => removeQuestion(i)}
                disabled={disabled}
                className="rounded-sm border border-alert/40 px-2 py-1 font-mono text-2xs uppercase tracking-wider text-alert transition-colors hover:bg-alert/10 disabled:opacity-50"
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
