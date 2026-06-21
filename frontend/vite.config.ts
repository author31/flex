/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173, // container port; host-published as 5000 (see docker-compose.yml)
    hmr: { clientPort: 5000 }, // HMR websocket reaches the dev server via the host port
    proxy: {
      // Dev: forward API calls to the backend container/service.
      "/api": { target: "http://backend:8000", changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
