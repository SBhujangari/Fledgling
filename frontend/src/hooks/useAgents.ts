import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { AgentResponse } from "@/types"

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => api.getAgents(),
  })
}

export function useRegisterAgent() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.registerAgent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] })
    },
  })
}

