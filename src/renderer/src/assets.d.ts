/** Images imported from src/renderer/src/assets arrive as URLs, which Vite fingerprints at build time. */
declare module '*.svg' {
  const src: string
  export default src
}
