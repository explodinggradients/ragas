import { ArrowLeft } from "lucide-react"

interface TitleHeaderProps {
  title: string
  showBackButton?: boolean
}

export function TitleHeader({ title, showBackButton }: TitleHeaderProps) {
  return (
    <div className="flex items-center gap-2">
      {showBackButton && (
        <button 
          onClick={() => window.history.back()} 
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </button>
      )}
      <h1 className="text-2xl font-semibold">{title}</h1>
    </div>
  )
}