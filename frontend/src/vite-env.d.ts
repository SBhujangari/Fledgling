/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  // Add other env variables as needed
}

// Extend the existing ImportMetaEnv from vite/client
declare module 'vite/client' {
  interface ImportMetaEnv {
    readonly VITE_API_BASE_URL?: string
  }
}

