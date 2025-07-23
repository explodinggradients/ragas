import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: parseInt(process.env.PORT || '3000'),
    host: process.env.HOST || '127.0.0.1',
  },
  build: {
    outDir: '../js-bundle',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Ensure consistent file names for easier serving
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    }
  }
})
