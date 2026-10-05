import { defineConfig } from 'vite';

/**
 * Aetheria build configuration.
 *
 * Deliberately dependency-free: no WGSL plugin, no GLSL plugin. Every shader is
 * authored as a template literal in TypeScript so that shader source and the
 * uniform layouts that feed it live in the same module and cannot drift apart.
 */
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    // Single self-contained bundle keeps cold-start under the 3s budget and
    // avoids a waterfall of module requests on mobile Safari.
    cssCodeSplit: false,
    assetsInlineLimit: 0,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 2048,
    rollupOptions: {
      output: {
        // One self-contained bundle: a single request, no module waterfall, and
        // no risk of a chunk fetch failing after the canvas is already live.
        codeSplitting: false,
      },
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
  },
  preview: {
    port: 4173,
    host: '127.0.0.1',
  },
});
