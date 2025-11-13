import { Link, useLocation } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Zap, User, Settings } from "lucide-react"

export function Navigation() {
  const location = useLocation()

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + "/")
  }

  return (
    <nav className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 w-full">
      <div className="w-full px-4 sm:px-6 lg:px-8 flex h-16 items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <Zap className="size-6 text-primary" />
            <span className="text-xl font-semibold text-foreground">Fledgling</span>
          </Link>
          <div className="hidden items-center gap-6 md:flex">
            {[
              { path: "/dashboard", label: "Dashboard" },
              { path: "/playground", label: "Playground" },
              { path: "/traces", label: "Traces" },
              { path: "/ops", label: "Ops Console" },
              { path: "/metrics", label: "Metrics" },
              { path: "/slm-dashboard", label: "SLM Demo" },
            ].map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`text-sm font-medium transition-colors ${
                  isActive(link.path) ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon">
            <User className="size-5" />
          </Button>
          <Button variant="ghost" size="icon">
            <Settings className="size-5" />
          </Button>
        </div>
      </div>
    </nav>
  )
}
