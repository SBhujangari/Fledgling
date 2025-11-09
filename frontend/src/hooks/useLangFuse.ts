import { useMutation } from "@tanstack/react-query"
import { api } from "@/lib/api"

export function useConnectLangFuse() {
  return useMutation({
    mutationFn: api.connectLangFuse,
  })
}

