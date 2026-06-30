import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/stream": { target: "ws://localhost:8000", ws: true },
      "/health": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
    },
  },
});
