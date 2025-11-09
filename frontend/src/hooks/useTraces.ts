import { useQuery, useMutation } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { TracesResponse } from "@/types"

export function useTraces(params?: { updatedAfter?: string }) {
  return useQuery({
    queryKey: ["traces", params],
    queryFn: () => api.getTraces(params),
  })
}

export function useTrain() {
  return useMutation({
    mutationFn: api.train,
  })
}

