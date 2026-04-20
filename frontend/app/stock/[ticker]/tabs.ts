export interface TabDef {
  key: string;
  label: string;
  icon?: string;
}

export const TABS: TabDef[] = [
  { key: "overview",     label: "Overview",     icon: "📊" },
  { key: "technicals",   label: "Technicals",   icon: "📐" },
  { key: "sentiment",    label: "Sentiment",    icon: "📰" },
  { key: "fundamentals", label: "Fundamentals", icon: "🏦" },
  { key: "fii-dii",      label: "FII / DII",    icon: "💼" },
  { key: "patterns",     label: "Patterns",     icon: "🔍" },
  { key: "backtest",     label: "Backtest",     icon: "🛠️" },
  { key: "accuracy",     label: "Accuracy",     icon: "🎯" },
];

export const TAB_KEYS = new Set(TABS.map((t) => t.key));
export const DEFAULT_TAB = "overview";
