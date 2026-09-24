export type HistoryItem = {
  id: string;
  provider: "miso";
  label: string;
  createdAt: string;
};
export function misoHistory(value: unknown): HistoryItem[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (v): v is HistoryItem =>
      v &&
      typeof v === "object" &&
      v.provider === "miso" &&
      typeof v.id === "string" &&
      /^wrun_[a-zA-Z0-9_-]+$/.test(v.id) &&
      typeof v.label === "string" &&
      typeof v.createdAt === "string",
  );
}
