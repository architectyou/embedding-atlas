export function stemmer(s: string): string {
  // Fallback no-op stemmer to avoid external dependency during constrained builds.
  // For production, prefer the real 'stemmer' package.
  return s;
}

