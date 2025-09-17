import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import wasm from "vite-plugin-wasm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function fixAbsoluteImport() {
  // Fix a bug where vite outputs absolute paths for workers.
  return {
    name: "fix-absolute-import",
    renderChunk(code) {
      // new URL(/* @vite-ignore */ "/assets/worker_main-DWGFbKCZ.js"
      // ->
      // new URL("./assets/worker_main-DWGFbKCZ.js"
      return code.replace(/new URL\((\/\*.*?\*\/ *)?"\//g, `new URL("./`);
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte(), wasm(), dts({ rollupTypes: true }), fixAbsoluteImport()],
  worker: {
    plugins: () => [wasm()],
    format: "es",
    rollupOptions: {
      output: {
        entryFileNames: "[name].js",
      },
    },
  },
  build: {
    target: "esnext",
    lib: {
      entry: {
        index: "./src/lib/index.ts",
      },
      fileName: (_, entryName) => `${entryName}.js`,
      formats: ["es"],
    },
    rollupOptions: {
      external: ["@uwdata/mosaic-core", "@uwdata/mosaic-sql"],
    },
    copyPublicDir: false,
  },
  resolve: {
    alias: [
      { find: /(.*\/worker_functions)\.js$/, replacement: "$1.ts" },
      {
        find: "@embedding-atlas/density-clustering",
        replacement: path.resolve(
          __dirname,
          "../density-clustering/density_clustering_wasm/js/index.js",
        ),
      },
      {
        find: "stemmer",
        replacement: path.resolve(__dirname, "./src/vendor/stemmer.ts"),
      },
    ],
  },
  optimizeDeps: {
    exclude: ["svelte"],
  },
});
