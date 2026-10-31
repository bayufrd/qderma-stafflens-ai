import http from 'node:http'
import vue from '@vitejs/plugin-vue'
import { defineConfig, loadEnv } from 'vite'

const keepAliveAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 16,
  maxFreeSockets: 8,
  timeout: 60000,
  keepAliveMsecs: 30000,
})

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', 'DASHBOARD_')
  const port = Number(env.DASHBOARD_PORT || 8000)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('DASHBOARD_PORT must be between 1 and 65535')
  }
  return {
    plugins: [vue()],
    server: {
      host: '127.0.0.1',
      proxy: {
        '/api': {
          target: `http://127.0.0.1:${port}`,
          changeOrigin: false,
          ws: true,
          agent: keepAliveAgent,
        },
      },
    },
  }
})
