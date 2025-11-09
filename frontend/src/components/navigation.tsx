import { Link, useLocation } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Zap } from "lucide-react"

export function Navigation() {
  const location = useLocation()

  const isActive = (path: string) => {
    if (path === "/") {
      return location.pathname === "/"
    }
    return location.pathname.startsWith(path)
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
            <Link 
              to="/" 
              className={`text-sm font-medium transition-colors ${
                isActive("/") 
                  ? "text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Dashboard
            </Link>
            <Link
              to="/testing"
              className={`text-sm font-medium transition-colors ${
                isActive("/testing") 
                  ? "text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Testing
            </Link>
            <Link
              to="/tuning"
              className={`text-sm font-medium transition-colors ${
                isActive("/tuning") 
                  ? "text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Tuning
            </Link>
          </div>
        </div>
        <Button variant="outline" size="sm">
          Settings
        </Button>
      </div>
    </nav>
  )
}

