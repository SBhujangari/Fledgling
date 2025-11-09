import { useState } from "react"
import { Navigation } from "@/components/navigation"
import { OnboardingState } from "@/components/onboarding-state"
import { DashboardState } from "@/components/dashboard-state"

export default function HomePage() {
  // START: Hard-coded value - replace with backend API call to check if LangFuse is connected
  const [isConnected, setIsConnected] = useState(true)
  // END: Hard-coded value

  return (
    <div className="min-h-screen w-full h-full bg-background">
      <Navigation />
      {isConnected ? <DashboardState /> : <OnboardingState onComplete={() => setIsConnected(true)} />}
    </div>
  )
}

